#!/usr/bin/env python3
import argparse
import asyncio
import http
import json
import logging
import os
import socket
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import websockets
import websockets.asyncio.server
import websockets.frames
from PIL import Image

from deployment.model_server.tools import msgpack_numpy
from examples.Behavior.model2behavior_interface import M1Inference
from starVLA.dataloader.gr00t_lerobot.behavior1k_utils import (
    B1K_ACTION23_SLICES,
    B1K_DELTA_PREFIX_DIMS,
    B1K_ORDERED_ACTION_KEYS,
    extract_b1k_state_from_proprio,
    get_b1k_delta_action_slice,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import read_mode_config


def load_task_description(task_name: str, tasks_jsonl_path: Path) -> str:
    with open(tasks_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            task_data = json.loads(line)
            if task_data["task_name"] == task_name:
                return task_data["task"]
    raise KeyError(f"Task name '{task_name}' not found in {tasks_jsonl_path}")


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value


def _tensor_stats(name: str, value: Any) -> str:
    array = _as_numpy(value)
    if array is None:
        return f"{name}: None"
    if array.size == 0:
        return f"{name}: shape={array.shape}, dtype={array.dtype}, empty"
    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        array_float = array.astype(np.float32, copy=False)
        return (
            f"{name}: shape={array.shape}, dtype={array.dtype}, "
            f"min={float(np.min(array_float)):.6f}, "
            f"max={float(np.max(array_float)):.6f}, "
            f"mean={float(np.mean(array_float)):.6f}"
        )
    return f"{name}: shape={array.shape}, dtype={array.dtype}"


def _format_action_vector(name: str, value: Any) -> str:
    array = _as_numpy(value)
    if array is None:
        return f"{name}: None"
    if array.size == 0:
        return f"{name}: shape={array.shape}, dtype={array.dtype}, empty"
    flat = array.reshape(-1)
    return (
        f"{name}: shape={array.shape}, dtype={array.dtype}, values="
        f"{np.array2string(flat, precision=5, separator=', ', max_line_width=100000)}"
    )


def _get_b1k_delta_prefix_dim(action_key: str, action_dim: int, qpos_dim: int) -> int:
    expected_action_slice = B1K_ACTION23_SLICES.get(action_key)
    expected_action_dim = 0 if expected_action_slice is None else expected_action_slice.stop - expected_action_slice.start
    if action_dim != expected_action_dim:
        raise ValueError(
            f"Unexpected B1K action width for {action_key}: got {action_dim}, expected {expected_action_dim}."
        )
    if qpos_dim != action_dim:
        raise ValueError(
            f"B1K action/state slice width mismatch for {action_key}: action_dim={action_dim}, qpos_dim={qpos_dim}."
        )
    delta_dim = B1K_DELTA_PREFIX_DIMS.get(action_key, 0)
    if delta_dim > action_dim:
        raise ValueError(
            f"Invalid B1K delta layout for {action_key}: delta_dim={delta_dim}, action_dim={action_dim}."
        )
    return delta_dim


def _load_episode_actions(parquet_path: Path) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Episode replay mode requires pandas in the serving environment. "
            "Please install pandas before using --replay-episode-parquet."
        ) from exc

    dataframe = pd.read_parquet(parquet_path)
    if "action" not in dataframe.columns:
        raise KeyError(f"'action' column not found in replay episode parquet: {parquet_path}")

    actions = np.asarray(dataframe["action"].tolist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 23:
        raise ValueError(f"Replay episode actions must have shape [T, 23], got {actions.shape} from {parquet_path}")
    return actions


def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32, copy=False)
    if array.size > 0 and float(np.max(array)) <= 1.0 and float(np.min(array)) >= 0.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _save_debug_images(example: Dict[str, Any], save_dir: Path, prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(example.get("image") or []):
        array = _as_numpy(image)
        if array is None or array.ndim != 3:
            continue
        image_path = save_dir / f"{prefix}_image{idx}.png"
        Image.fromarray(_to_uint8_image(array)).save(image_path)
        logging.info("Saved model input image[%d] to %s", idx, image_path)

    wrist_views = example.get("wrist_views")
    if wrist_views is None:
        return
    for idx, image in enumerate(wrist_views):
        array = _as_numpy(image)
        if array is None or array.ndim != 3:
            continue
        image_path = save_dir / f"{prefix}_wrist{idx}.png"
        Image.fromarray(_to_uint8_image(array)).save(image_path)
        logging.info("Saved model wrist_view[%d] to %s", idx, image_path)


def _log_example_io(
    example: Dict[str, Any],
    normalized_actions: Any,
    raw_actions: Any,
    final_action: Any,
    save_dir: Optional[Path] = None,
    save_prefix: Optional[str] = None,
) -> None:
    logging.info("Model prompt: %s", example["lang"])
    if save_dir is not None and save_prefix is not None:
        _save_debug_images(example, save_dir, save_prefix)
    for idx, image in enumerate(example.get("image") or []):
        logging.info(_tensor_stats(f"image[{idx}]", image))
    wrist_views = example.get("wrist_views")
    if wrist_views is None:
        logging.info("wrist_views: None")
    else:
        for idx, image in enumerate(wrist_views):
            logging.info(_tensor_stats(f"wrist_views[{idx}]", image))
    if "state" in example:
        logging.info(_tensor_stats("state", example["state"]))
    logging.info(_tensor_stats("normalized_actions", normalized_actions))
    logging.info(_tensor_stats("raw_actions", raw_actions))
    logging.info(_tensor_stats("final_action", final_action))


def _resolve_stats_path(path_str: str, policy_ckpt_path: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = Path(policy_ckpt_path).expanduser().resolve()
    search_roots = [
        Path.cwd(),
        repo_root,
        ckpt_path.parent,
        ckpt_path.parent.parent,
    ]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (repo_root / candidate).resolve()


def _load_behavior_skill_action_stats(policy_ckpt_path: str) -> dict[str, np.ndarray]:
    cfg, _ = read_mode_config(policy_ckpt_path)
    cfg = cfg or {}
    vla_data_cfg = (cfg.get("datasets") or {}).get("vla_data") or {}
    action_norm_stats_path = vla_data_cfg.get("action_norm_stats_path")
    if not action_norm_stats_path:
        raise ValueError(
            "Behavior-Skill eval requires datasets.vla_data.action_norm_stats_path in checkpoint config."
        )

    stats_path = _resolve_stats_path(str(action_norm_stats_path), policy_ckpt_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Behavior-Skill norm stats file not found: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "norm_stats" in payload and isinstance(payload["norm_stats"], dict):
        payload = payload["norm_stats"]

    action_block = payload.get("actions") or payload.get("action")
    if not isinstance(action_block, dict):
        raise ValueError("Behavior-Skill norm stats must contain an `actions` or `action` block.")

    required_names = ["q01", "q99", "per_timestamp_q01", "per_timestamp_q99"]
    for stat_name in required_names:
        if stat_name not in action_block:
            raise ValueError(f"Behavior-Skill eval requires `{stat_name}` in external norm stats.")

    stats = {}
    for stat_name in required_names:
        array = np.asarray(action_block[stat_name], dtype=np.float32)
        if stat_name.startswith("per_timestamp_"):
            if array.ndim != 2:
                raise ValueError(f"{stat_name} must be a 2D array, got shape={array.shape}")
            if array.shape[1] < 23:
                raise ValueError(f"{stat_name} width must be >= 23, got shape={array.shape}")
            ordered = np.zeros((array.shape[0], 23), dtype=np.float32)
            for action_key in B1K_ORDERED_ACTION_KEYS:
                action_slice = B1K_ACTION23_SLICES[action_key]
                ordered[:, action_slice] = array[:, action_slice]
        else:
            if array.ndim != 1:
                raise ValueError(f"{stat_name} must be a 1D array, got shape={array.shape}")
            if array.shape[0] < 23:
                raise ValueError(f"{stat_name} width must be >= 23, got shape={array.shape}")
            ordered = np.zeros((23,), dtype=np.float32)
            for action_key in B1K_ORDERED_ACTION_KEYS:
                action_slice = B1K_ACTION23_SLICES[action_key]
                ordered[action_slice] = array[action_slice]
        stats[stat_name] = ordered

    logging.info("Loaded Behavior-Skill q99 stats from %s", stats_path)
    return stats


def _unnormalize_behavior_q99_actions(normalized_actions: np.ndarray, action_norm_stats: dict[str, np.ndarray]) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    if normalized_actions.ndim != 2 or normalized_actions.shape[1] != 23:
        raise ValueError(f"Expected normalized actions with shape [T, 23], got {normalized_actions.shape}")

    normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
    steps = normalized_actions.shape[0]

    q01 = action_norm_stats["per_timestamp_q01"][:steps]
    q99 = action_norm_stats["per_timestamp_q99"][:steps]
    if q01.shape != normalized_actions.shape or q99.shape != normalized_actions.shape:
        raise ValueError(
            "Per-time q99 stats shape mismatch: "
            f"actions={normalized_actions.shape}, q01={q01.shape}, q99={q99.shape}"
        )

    # Keep the same epsilon as the training-time q99 normalization
    # (BehaviorQ99PerTimeTransform uses `q99 - q01 + 1e-6`) so the eval-time
    # inverse transform stays numerically consistent with training.
    return 0.5 * (normalized_actions + 1.0) * (q99 - q01 + 1e-6) + q01


class OmniGibsonCompatServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata: dict | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = msgpack_numpy.unpackb(await websocket.recv())
                if "reset" in result:
                    self._policy.reset()
                    continue

                infer_time = time.monotonic()
                action = self._policy.act(result)
                infer_time = time.monotonic() - infer_time

                payload = {
                    "action": action.detach().cpu().numpy(),
                    "server_timing": {"infer_ms": infer_time * 1000},
                }
                if prev_total_time is not None:
                    payload["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                if getattr(self._policy, "debug_model_io", False):
                    logging.info(_format_action_vector("payload.action", payload["action"]))
                await websocket.send(packer.pack(payload))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logging.error("Error in connection from %s:\n%s", websocket.remote_address, traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


class BehaviorSkillEvalCompatPolicy(M1Inference):
    def __init__(
        self,
        policy_ckpt_path: str,
        task_description: str,
        policy_setup: str = "R1Pro",
        image_size: list[int] = [224, 224],
        use_bf16: bool = False,
        use_state: bool = False,
        execute_in_n_steps: int = 20,
        model_rollout_start_index: int = 0,
        debug_model_io: bool = False,
        replay_episode_parquet: Optional[str] = None,
        replay_episode_loop: bool = False,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if policy_setup != "R1Pro":
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for BEHAVIOR models.")

        self.policy_setup = policy_setup
        self.unnorm_key = "R1Pro"
        self.action_ensemble = False
        self.policy_ckpt_path = policy_ckpt_path
        self.use_ddim = True
        self.num_ddim_steps = 10
        self.cfg_scale = 1.5
        self.image_size = image_size
        self.action_scale = 1.0
        self.horizon = 0
        self.task_description = task_description
        self.requested_use_state = use_state
        self.model_uses_state = False
        self.debug_model_io = debug_model_io
        self.replay_episode_path = Path(replay_episode_parquet).expanduser().resolve() if replay_episode_parquet else None
        self.replay_episode_loop = replay_episode_loop

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.execute_in_n_steps = execute_in_n_steps
        self.model_rollout_start_index = int(model_rollout_start_index)
        if self.model_rollout_start_index < 0:
            raise ValueError(
                f"model_rollout_start_index must be >= 0, got {self.model_rollout_start_index}"
            )
        self.last_actions = None
        self.action_index = 0
        self.prediction_count = 0
        self.current_step = 0
        self.last_normalized_actions = None
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0
        self.debug_image_dir = Path(policy_ckpt_path).resolve().parent / "server_logs"
        self.replay_actions = None
        self.replay_cursor = 0
        self.replay_exhausted = False
        self.action_mode = "abs"
        self.action_norm_mode = "q99"
        self.action_mode_apply_keys = ["action.torso", "action.left_arm", "action.right_arm"]

        logging.info("Using fixed execution window: %d env steps per model rollout", self.execute_in_n_steps)
        logging.info("Using model rollout start index: %d", self.model_rollout_start_index)

        if self.replay_episode_path is not None:
            self.replay_actions = _load_episode_actions(self.replay_episode_path)
            logging.info(
                "Using replay episode actions from %s, total_steps=%d, loop=%s",
                self.replay_episode_path,
                len(self.replay_actions),
                self.replay_episode_loop,
            )
            self.action_norm_stats = None
            self.vla = None
        else:
            self._load_eval_action_mode_config(policy_ckpt_path)
            self.action_norm_stats = _load_behavior_skill_action_stats(policy_ckpt_path)
            self.vla = baseframework.from_pretrained(policy_ckpt_path)
            if use_bf16:
                self.vla = self.vla.to(torch.bfloat16)
            self.vla = self.vla.to("cuda").eval()

    def reset(self, task_description: Optional[str] = None) -> None:
        super().reset(task_description=task_description)
        self.last_actions = None
        self.action_index = 0
        self.prediction_count = 0
        self.last_normalized_actions = None
        self.replay_cursor = 0
        self.replay_exhausted = False

    def _load_eval_action_mode_config(self, policy_ckpt_path: str) -> None:
        cfg, _ = read_mode_config(policy_ckpt_path)
        cfg = cfg or {}
        framework_cfg = cfg.get("framework") or {}
        vla_data_cfg = (cfg.get("datasets") or {}).get("vla_data") or {}

        self.action_mode = str(vla_data_cfg.get("action_mode", "abs")).lower()
        if self.action_mode != "b1k_delta":
            raise ValueError(
                f"Behavior-Skill eval expects action_mode='b1k_delta', got {self.action_mode!r}"
            )

        self.action_mode_apply_keys = list(
            vla_data_cfg.get("action_mode_apply_keys", ["action.torso", "action.left_arm", "action.right_arm"])
        )

        include_state = vla_data_cfg.get("include_state", False)
        if isinstance(include_state, str):
            include_state = include_state.strip().lower() not in {"", "0", "false", "no", "off"}
        self.model_uses_state = bool(include_state)
        state_injection_mode = str(framework_cfg.get("state_injection_mode", "prompt")).lower()

        if self.requested_use_state != self.model_uses_state:
            logging.info(
                "Ignoring CLI --use-state=%s; checkpoint config requires model_uses_state=%s",
                self.requested_use_state,
                self.model_uses_state,
            )
        logging.info(
            "Loaded Behavior-Skill eval config: action_mode=%s, action_norm_mode=%s, model_uses_state=%s, state_injection_mode=%s",
            self.action_mode,
            self.action_norm_mode,
            self.model_uses_state,
            state_injection_mode,
        )

    def _build_b1k_state_ref_23(self, raw_proprio: np.ndarray) -> np.ndarray:
        raw_proprio = np.asarray(raw_proprio, dtype=np.float32)
        if raw_proprio.ndim != 1:
            raise ValueError(f"Expected 1D proprio for b1k delta decode, got shape={raw_proprio.shape}")
        return extract_b1k_state_from_proprio(raw_proprio)

    def _maybe_decode_action_mode(self, raw_actions: np.ndarray, raw_proprio: Optional[np.ndarray]) -> np.ndarray:
        if self.action_mode != "b1k_delta":
            return raw_actions
        if raw_proprio is None:
            raise ValueError("action_mode='b1k_delta' requires raw proprio during evaluation.")

        state_ref = self._build_b1k_state_ref_23(raw_proprio)
        decoded = np.array(raw_actions, copy=True, dtype=np.float32)
        for action_key in self.action_mode_apply_keys:
            action_slice = B1K_ACTION23_SLICES.get(action_key)
            if action_slice is None:
                raise ValueError(f"Unsupported B1K action key in action_mode_apply_keys: {action_key}")
            delta_dim = _get_b1k_delta_prefix_dim(
                action_key=action_key,
                action_dim=action_slice.stop - action_slice.start,
                qpos_dim=action_slice.stop - action_slice.start,
            )
            if delta_dim <= 0:
                continue
            delta_slice = get_b1k_delta_action_slice(action_key)
            if delta_slice is None:
                continue
            decoded[:, delta_slice] += state_ref[delta_slice]
        return decoded

    def _next_replay_rollout(self) -> np.ndarray:
        assert self.replay_actions is not None
        if len(self.replay_actions) == 0:
            raise ValueError(f"No actions found in replay episode: {self.replay_episode_path}")

        rollout_start = self.replay_cursor
        rollout_size = self.execute_in_n_steps
        if self.replay_episode_loop:
            indices = [(rollout_start + offset) % len(self.replay_actions) for offset in range(rollout_size)]
            self.replay_cursor = (rollout_start + rollout_size) % len(self.replay_actions)
            return self.replay_actions[indices].copy()

        if rollout_start >= len(self.replay_actions):
            if not self.replay_exhausted:
                logging.warning(
                    "Replay episode %s exhausted at step %d; reusing the final action for subsequent requests.",
                    self.replay_episode_path,
                    rollout_start,
                )
                self.replay_exhausted = True
            return self.replay_actions[-1:].copy()

        rollout_end = min(rollout_start + rollout_size, len(self.replay_actions))
        self.replay_cursor = rollout_end
        return self.replay_actions[rollout_start:rollout_end].copy()

    def act(self, obs: Dict[str, Any]) -> torch.Tensor:
        processed_obs = self._process_behavior_obs(obs)

        primary_image = processed_obs["full_image"]
        left_wrist_image = processed_obs["left_wrist_image"]
        right_wrist_image = processed_obs["right_wrist_image"]
        if "dual" in self.policy_ckpt_path.lower():
            image_input = [primary_image]
            wrist_image_input = [left_wrist_image, right_wrist_image]
        else:
            image_input = [primary_image, left_wrist_image, right_wrist_image]
            wrist_image_input = None

        raw_state = processed_obs["state"]
        raw_proprio = np.asarray(obs["robot_r1::proprio"], dtype=np.float32)
        if self.model_uses_state:
            example = {
                "image": image_input,
                "wrist_views": wrist_image_input,
                "state": raw_state[None, :],
                "lang": self.task_description,
            }
        else:
            example = {
                "image": image_input,
                "wrist_views": wrist_image_input,
                "lang": self.task_description,
            }

        if self.last_actions is None or self.action_index >= len(self.last_actions):
            prev_executed_last = None if self.last_actions is None else np.array(self.last_actions[-1], copy=True)
            rollout_start = self.replay_cursor
            if self.replay_actions is None:
                with torch.no_grad():
                    response = self.vla.predict_action(examples=[example])
                normalized_actions = response["normalized_actions"][0]
                unnormalized_actions = _unnormalize_behavior_q99_actions(normalized_actions, self.action_norm_stats)
                decoded_actions = self._maybe_decode_action_mode(unnormalized_actions, raw_proprio)
                rollout_start_index = self.model_rollout_start_index
                if rollout_start_index >= len(decoded_actions):
                    raise ValueError(
                        "model_rollout_start_index=%d is out of range for decoded_actions length=%d"
                        % (rollout_start_index, len(decoded_actions))
                    )
                actions_to_execute = min(len(decoded_actions) - rollout_start_index, self.execute_in_n_steps)
                self.last_actions = decoded_actions[rollout_start_index : rollout_start_index + actions_to_execute].copy()
                self.last_normalized_actions = normalized_actions
            else:
                self.last_actions = self._next_replay_rollout()
                self.last_normalized_actions = None
                actions_to_execute = len(self.last_actions)
                rollout_start_index = 0
                unnormalized_actions = None
                decoded_actions = self.last_actions
            self.action_index = 0
            self.prediction_count += 1

            if self.replay_actions is None:
                logging.info(
                    "Prediction #%d generated %d actions, executing decoded actions [%d:%d)",
                    self.prediction_count,
                    len(decoded_actions),
                    rollout_start_index,
                    rollout_start_index + actions_to_execute,
                )
            else:
                logging.info(
                    "Replay rollout #%d prepared %d actions from episode step %d",
                    self.prediction_count,
                    actions_to_execute,
                    rollout_start,
                )

            if self.debug_model_io:
                rollout_kind = "Replay" if self.replay_actions is not None else "Model"
                logging.info("===== %s Rollout Debug #%d =====", rollout_kind, self.prediction_count)
                debug_rollout_actions = self.last_actions if self.replay_actions is not None else unnormalized_actions
                _log_example_io(
                    example,
                    self.last_normalized_actions,
                    debug_rollout_actions,
                    self.last_actions[0],
                    save_dir=self.debug_image_dir,
                    save_prefix=f"rollout_{self.prediction_count:04d}",
                )
                if self.last_normalized_actions is not None:
                    logging.info(_format_action_vector("normalized_actions[0]", self.last_normalized_actions[0]))
                    logging.info(_format_action_vector("normalized_actions[-1]", self.last_normalized_actions[-1]))
                if self.replay_actions is None:
                    logging.info("executed_rollout_source_start_index=%d", rollout_start_index)
                    logging.info(_format_action_vector("unnormalized_actions[0]", unnormalized_actions[0]))
                    logging.info(_format_action_vector("unnormalized_actions[-1]", unnormalized_actions[-1]))
                    if rollout_start_index > 0:
                        logging.info(
                            _format_action_vector(
                                f"unnormalized_actions[{rollout_start_index}]",
                                unnormalized_actions[rollout_start_index],
                            )
                        )
                    if prev_executed_last is not None:
                        logging.info(_format_action_vector("prev_executed_last", prev_executed_last))
                    state_ref_23 = self._build_b1k_state_ref_23(raw_proprio)
                    logging.info(_format_action_vector("rollout_raw_proprio", raw_proprio))
                    logging.info(_format_action_vector("rollout_raw_state", raw_state))
                    logging.info(_format_action_vector("rollout_state_ref_23", state_ref_23))
                    logging.info(_format_action_vector("decoded_actions[0]", decoded_actions[0]))
                    logging.info(_format_action_vector("decoded_actions[-1]", decoded_actions[-1]))
                    if prev_executed_last is not None:
                        logging.info(_format_action_vector("boundary_gap_decoded0", decoded_actions[0] - prev_executed_last))

        raw_actions = self.last_actions[self.action_index][None]
        self.action_index += 1
        self.current_step += 1

        raw_action = {
            "base_pose": np.array(raw_actions[0, :3]),
            "torso_pose": np.array(raw_actions[0, 3:7]),
            "left_arm_pose": np.array(raw_actions[0, 7:14]),
            "left_gripper_pose": np.array(raw_actions[0, 14:15]),
            "right_arm_pose": np.array(raw_actions[0, 15:22]),
            "right_gripper_pose": np.array(raw_actions[0, 22:23]),
        }
        action = self._process_action_for_behavior(raw_action)
        if self.debug_model_io:
            logging.info(
                "step=%d action_index=%d/%d source=%s %s",
                self.current_step,
                self.action_index,
                len(self.last_actions),
                "replay_episode" if self.replay_actions is not None else "model",
                _tensor_stats("final_action", action),
            )
            logging.info(_format_action_vector("raw_action", raw_actions[0]))
            logging.info(_format_action_vector("behavior_action", action))
        return torch.from_numpy(action).to(torch.float32)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--behavior-tasks-jsonl-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--policy-setup", type=str, default="R1Pro")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-state", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--execute-in-n-steps", type=int, default=20)
    parser.add_argument("--model-rollout-start-index", type=int, default=0)
    parser.add_argument("--debug-model-io", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--replay-episode-parquet", type=str, default=None)
    parser.add_argument("--replay-episode-loop", type=lambda x: str(x).lower() == "true", default=False)
    return parser


def main(args) -> None:
    if not args.ckpt_path and not args.replay_episode_parquet:
        raise ValueError("Please provide --ckpt_path for model serving or --replay-episode-parquet for episode replay.")
    task_description = load_task_description(args.task_name, Path(args.behavior_tasks_jsonl_path))
    policy = BehaviorSkillEvalCompatPolicy(
        policy_ckpt_path=args.ckpt_path or args.replay_episode_parquet,
        task_description=task_description,
        policy_setup=args.policy_setup,
        use_bf16=args.use_bf16,
        use_state=args.use_state,
        execute_in_n_steps=args.execute_in_n_steps,
        model_rollout_start_index=args.model_rollout_start_index,
        debug_model_io=args.debug_model_io,
        replay_episode_parquet=args.replay_episode_parquet,
        replay_episode_loop=args.replay_episode_loop,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
    logging.info("Creating Behavior-Skill OmniGibson-compatible server (host: %s, ip: %s)", hostname, local_ip)

    server = OmniGibsonCompatServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={"env": "behavior_skill_eval_compat", "task_name": args.task_name},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
