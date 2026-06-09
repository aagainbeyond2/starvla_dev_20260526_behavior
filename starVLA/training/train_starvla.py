# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.experiment_logger import ExperimentLogger
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def resolve_action_logging_spec(cfg) -> Dict | None:
    """Resolve action-dimension names and groups from the current VLA mixture."""
    try:
        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

        data_mix = cfg.datasets.vla_data.data_mix
        mixture_spec = DATASET_NAMED_MIXTURES.get(data_mix)
        if not mixture_spec:
            return None

        robot_types = {robot_type for _, _, robot_type in mixture_spec}
        if len(robot_types) != 1:
            logger.warning(f"Skip action logging spec: data_mix `{data_mix}` contains multiple robot types.")
            return None

        robot_type = next(iter(robot_types))
        data_config = ROBOT_TYPE_CONFIG_MAP.get(robot_type)
        if data_config is None:
            return None

        action_dim_names = getattr(data_config, "get_action_dim_names", lambda: None)()
        action_dim_groups = getattr(data_config, "get_action_dim_groups", lambda: None)()
        return {
            "robot_type": robot_type,
            "action_dim_names": list(action_dim_names) if action_dim_names else None,
            "action_dim_groups": dict(action_dim_groups) if action_dim_groups else None,
        }
    except Exception as exc:
        logger.warning(f"Failed to resolve action logging spec: {exc}")
        return None


def _to_numpy_debug(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.array(value)
    return None


def _sanitize_debug_key(key: str) -> str:
    return str(key).replace("/", "_").replace(".", "_").replace(" ", "_").replace(":", "_")


class TrainingDebugDumper:
    def __init__(self, cfg):
        debug_cfg = cfg.datasets.vla_data.get("debug_dump", {}) if cfg and cfg.datasets and cfg.datasets.vla_data else {}
        self.enabled = bool(debug_cfg.get("enabled", False))
        self.dump_every_steps = max(1, int(debug_cfg.get("dump_every_steps", 500)))
        self.max_dump_steps = max(0, int(debug_cfg.get("max_dump_steps", 5)))
        self.samples_per_step = max(1, int(debug_cfg.get("samples_per_step", 1)))
        self.log_action_rows = max(1, int(debug_cfg.get("log_action_rows", 2)))
        self.save_arrays = bool(debug_cfg.get("save_arrays", True))
        self.save_dir_name = str(debug_cfg.get("save_dir_name", "train_debug"))
        self.dumped_steps = 0
        self.dump_dir = Path(cfg.output_dir) / self.save_dir_name
        if self.enabled:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    def should_dump(self, step_index: int) -> bool:
        return (
            self.enabled
            and self.dumped_steps < self.max_dump_steps
            and step_index > 0
            and step_index % self.dump_every_steps == 0
        )

    def _resolve_dim_names(self, action_dim: int, action_logging_spec: Dict | None) -> list[str]:
        spec = action_logging_spec or {}
        dim_names = spec.get("action_dim_names") or [f"dim_{i:02d}" for i in range(action_dim)]
        dim_names = list(dim_names)[:action_dim]
        if len(dim_names) < action_dim:
            dim_names.extend(f"dim_{i:02d}" for i in range(len(dim_names), action_dim))
        return dim_names

    def _build_labeled_action_rows(self, action_target: np.ndarray, dim_names: list[str]) -> list[dict]:
        labeled_rows = []
        for row_idx, row in enumerate(action_target[: self.log_action_rows]):
            labeled_rows.append(
                {
                    "row_index": int(row_idx),
                    "values": {dim_names[dim_idx]: float(row[dim_idx]) for dim_idx in range(len(dim_names))},
                }
            )
        return labeled_rows

    def dump(self, step_index: int, batch_vla, debug_info: dict, model, action_logging_spec: Dict | None = None) -> None:
        if not self.should_dump(step_index):
            return

        step_dir = self.dump_dir / f"step_{step_index:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        action_loss_per_dim = debug_info.get("action_loss_per_dim")
        if isinstance(action_loss_per_dim, torch.Tensor):
            action_loss_per_dim = action_loss_per_dim.detach().float().cpu().tolist()

        for sample_idx, sample in enumerate(batch_vla[: self.samples_per_step]):
            sample_dir = step_dir / f"sample_{sample_idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            packed_action = np.asarray(sample["action"], dtype=np.float32)
            packed_state = np.asarray(sample["state"], dtype=np.float32) if "state" in sample else None
            action_horizon = int(getattr(model, "action_horizon", packed_action.shape[0]))
            action_target = packed_action[-action_horizon:]
            action_dim_names = self._resolve_dim_names(action_target.shape[-1], action_logging_spec)
            labeled_action_rows = self._build_labeled_action_rows(action_target, action_dim_names)

            summary = {
                "step_index": int(step_index),
                "sample_index": int(sample_idx),
                "robot_tag": sample.get("robot_tag"),
                "lang": sample.get("lang"),
                "action_loss": float(debug_info.get("action_loss", 0.0)),
                "action_loss_per_dim": action_loss_per_dim,
                "packed_action_shape": list(packed_action.shape),
                "packed_state_shape": list(packed_state.shape) if packed_state is not None else None,
                "action_target_shape": list(action_target.shape),
                "action_dim_names": action_dim_names,
                "action_target_rows": labeled_action_rows,
            }
            with open(sample_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(
                "[ActionDebug] step=%d sample=%d action_target_rows=%s",
                step_index,
                sample_idx,
                json.dumps(labeled_action_rows, ensure_ascii=False),
            )

            if self.save_arrays:
                flat_arrays = {
                    "packed_action": packed_action,
                    "action_target": action_target,
                }
                if packed_state is not None:
                    flat_arrays["packed_state"] = packed_state
                np.savez_compressed(sample_dir / "debug_arrays.npz", **flat_arrays)

        self.dumped_steps += 1
        logger.info(f"Saved training debug dump for step {step_index} to {step_dir}")


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.experiment_logger = ExperimentLogger(cfg=self.config, logger=logger)
        self.action_logging_spec = resolve_action_logging_spec(self.config)
        self.debug_dumper = TrainingDebugDumper(self.config)

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self.experiment_logger.init()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            self.experiment_logger.log_metrics(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _append_action_dim_metrics(self, metrics, output_dict):
        """Flatten per-dimension action losses into scalar logger metrics."""
        loss_per_dim = output_dict.get("action_loss_per_dim")
        if loss_per_dim is None:
            return

        if isinstance(loss_per_dim, torch.Tensor):
            loss_per_dim = loss_per_dim.detach().float().cpu().tolist()
        else:
            loss_per_dim = [float(v) for v in loss_per_dim]

        spec = self.action_logging_spec or {}
        dim_names = spec.get("action_dim_names") or [f"dim_{i:02d}" for i in range(len(loss_per_dim))]
        dim_names = list(dim_names)[: len(loss_per_dim)]
        if len(dim_names) < len(loss_per_dim):
            dim_names.extend(f"dim_{i:02d}" for i in range(len(dim_names), len(loss_per_dim)))

        for idx, (name, value) in enumerate(zip(dim_names, loss_per_dim)):
            safe_name = str(name).replace("/", "_")
            metrics[f"action_dit_loss/dim/{idx:02d}_{safe_name}"] = float(value)

        group_slices = spec.get("action_dim_groups") or {}
        for group_name, span in group_slices.items():
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start < 0 or end > len(loss_per_dim) or start >= end:
                continue
            group_values = loss_per_dim[start:end]
            metrics[f"action_dit_loss/group/{group_name}"] = float(sum(group_values) / len(group_values))

    def _append_eval_action_l1_metrics(self, metrics, predicted_actions, gt_actions):
        """Log per-dimension eval L1 averaged over batch and time."""
        abs_error = np.abs(predicted_actions - gt_actions)
        metrics["eval/action_l1_mean"] = float(abs_error.mean())

        l1_per_dim = abs_error.mean(axis=(0, 1))
        spec = self.action_logging_spec or {}
        dim_names = spec.get("action_dim_names") or [f"dim_{i:02d}" for i in range(len(l1_per_dim))]
        dim_names = list(dim_names)[: len(l1_per_dim)]
        if len(dim_names) < len(l1_per_dim):
            dim_names.extend(f"dim_{i:02d}" for i in range(len(dim_names), len(l1_per_dim)))

        for idx, (name, value) in enumerate(zip(dim_names, l1_per_dim)):
            safe_name = str(name).replace("/", "_")
            metrics[f"eval/action_l1/dim/{idx:02d}_{safe_name}"] = float(value)

        group_slices = spec.get("action_dim_groups") or {}
        for group_name, span in group_slices.items():
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start < 0 or end > len(l1_per_dim) or start >= end:
                continue
            group_values = l1_per_dim[start:end]
            metrics[f"eval/action_l1/group/{group_name}"] = float(group_values.mean())

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics, debug_info = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            next_step_index = self.completed_steps + 1 if self.accelerator.sync_gradients else self.completed_steps
            if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                self.debug_dumper.dump(
                    step_index=next_step_index,
                    batch_vla=batch_vla,
                    debug_info=debug_info,
                    model=self.accelerator.unwrap_model(self.model),
                    action_logging_spec=self.action_logging_spec,
                )

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots
            self._append_eval_action_l1_metrics(step_metrics, normalized_actions, actions)

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        log_dict = {
            "action_dit_loss": action_loss.item(),
        }
        self._append_action_dim_metrics(log_dict, output_dict)
        return log_dict, {
            "action_loss": action_loss.item(),
            "action_loss_per_dim": output_dict.get("action_loss_per_dim"),
        }

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            self.experiment_logger.close()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
