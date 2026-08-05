import copy
import inspect
import json
import logging
import multiprocessing as mp
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Sampler

from starVLA.dataloader.gr00t_lerobot.behavior1k_utils import B1K_ACTION23_SLICES
from starVLA.dataloader.gr00t_lerobot.behavior_skill_dataset import BehaviorSkillSingleDataset
from starVLA.dataloader.gr00t_lerobot.datasets import (
    DatasetMetadata,
    LeRobotMixtureDataset,
    safe_hash,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.worker_resume_state import deterministic_sample_rng

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return batch


def _patch_behavior_skill_action_mask(stats_payload: dict) -> dict:
    """Restore the legacy gripper mask semantics for Behavior-Skill exports only."""
    for tag_stats in stats_payload.values():
        if not isinstance(tag_stats, dict):
            continue
        action_stats = tag_stats.get("action")
        if not isinstance(action_stats, dict):
            continue
        mask = action_stats.get("mask")
        if not isinstance(mask, list):
            continue

        patched_mask = list(mask)
        for action_key in ("action.left_gripper", "action.right_gripper"):
            action_slice = B1K_ACTION23_SLICES.get(action_key)
            if action_slice is None or action_slice.stop > len(patched_mask):
                continue
            for index in range(action_slice.start, action_slice.stop):
                patched_mask[index] = False
        action_stats["mask"] = patched_mask

    return stats_payload


class BehaviorSkillMixtureDataset(LeRobotMixtureDataset):
    """Behavior-only wrapper that propagates epoch updates to child datasets."""

    def __init__(self, *args, **kwargs):
        # Persistent DataLoader workers have private Python objects, but this
        # scalar remains shared with the trainer process across epoch updates.
        self._shared_epoch = mp.RawValue("q", 0)
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        self._shared_epoch.value = epoch
        super().set_epoch(epoch)
        for dataset in getattr(self, "datasets", []):
            if callable(getattr(dataset, "set_epoch", None)):
                dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        # Accelerate checkpoints the trainer RNG but not persistent worker RNG
        # streams. Seeding each sample makes a mid-epoch restart reproduce the
        # same mixture choice, retry path, and image augmentation parameters.
        epoch = int(self._shared_epoch.value)
        self.epoch = epoch
        sample_seed = safe_hash(("behavior-skill-sample", epoch, int(index), int(self.seed)))
        with deterministic_sample_rng(sample_seed):
            return super().__getitem__(index)

    def update_metadata(self, metadata_config: dict, cached_statistics_path=None) -> None:
        """Merge shared stats while retaining each child's raw video resolution."""
        video_configs = {
            json.dumps(
                dataset.metadata.model_dump(mode="json")["modalities"]["video"],
                sort_keys=True,
            )
            for dataset in self.datasets
        }
        if len(video_configs) == 1:
            return super().update_metadata(metadata_config, cached_statistics_path)

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata = {}
        grouped = {}
        for dataset in self.datasets:
            grouped.setdefault(dataset.tag, []).append(dataset.metadata)

        for tag, metadatas in grouped.items():
            canonical_video = metadatas[0].model_dump(mode="json")["modalities"]["video"]
            mergeable = []
            for metadata in metadatas:
                payload = metadata.model_dump(mode="json")
                payload["modalities"]["video"] = canonical_video
                mergeable.append(DatasetMetadata.model_validate(payload))
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=mergeable,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )

        for dataset in self.datasets:
            child_payload = self.merged_metadata[dataset.tag].model_dump(mode="json")
            child_payload["modalities"]["video"] = dataset.metadata.model_dump(
                mode="json"
            )["modalities"]["video"]
            dataset.set_transforms_metadata(
                DatasetMetadata.model_validate(child_payload)
            )

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        super().save_dataset_statistics(save_path, format=format)
        if format != "json":
            return

        save_path = Path(save_path)
        with open(save_path, "r", encoding="utf-8") as f:
            stats_payload = json.load(f)

        stats_payload = _patch_behavior_skill_action_mask(stats_payload)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)
        logger.info("[behavior-skill] Patched exported action mask for legacy gripper compatibility: %s", save_path)


class BehaviorSkillEpochSampler(Sampler[int]):
    """Epoch-aware shuffled sampler for Behavior-Skill training."""

    def __init__(self, dataset: BehaviorSkillMixtureDataset):
        self.dataset = dataset
        self.epoch = 0
        self._resume_sample_offset = 0
        self._resume_epoch = None
        self.dataset.set_epoch(0)

    def __iter__(self):
        # Match the trainer's epoch-reset flow: every epoch should see a new,
        # deterministic shuffle order instead of a fixed sequential scan.
        generator = torch.Generator()
        generator.manual_seed(int(self.epoch))
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(indices[self._resume_sample_offset :])

    def __len__(self) -> int:
        return max(0, len(self.dataset) - self._resume_sample_offset)

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        preserve_resume_offset = self._resume_epoch == epoch
        self.epoch = epoch
        if preserve_resume_offset:
            self.dataset.set_epoch(epoch)
            return
        self._resume_sample_offset = 0
        self._resume_epoch = None
        self.dataset.set_epoch(epoch)

    def set_resume_sample_offset(self, sample_offset: int):
        """Start the next iterator at a deterministic point within this epoch."""

        sample_offset = int(sample_offset)
        if sample_offset < 0 or sample_offset > len(self.dataset):
            raise ValueError(
                f"resume sample offset {sample_offset} is outside [0, {len(self.dataset)}]"
            )
        self._resume_sample_offset = sample_offset
        self._resume_epoch = self.epoch


def make_behavior_skill_dataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> BehaviorSkillSingleDataset:
    data_config = copy.deepcopy(ROBOT_TYPE_CONFIG_MAP[robot_type])

    modality_signature = inspect.signature(data_config.modality_config)
    if "data_cfg" in modality_signature.parameters:
        modality_config = data_config.modality_config(data_cfg=data_cfg)
    else:
        modality_config = data_config.modality_config()

    transform_signature = inspect.signature(data_config.transform)
    if "data_cfg" in transform_signature.parameters:
        transforms = data_config.transform(data_cfg=data_cfg)
    else:
        transforms = data_config.transform()

    dataset_path = Path(data_root_dir) / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        logger.warning(
            "DataConfig for robot_type=%r has no embodiment_tag, using %s as default",
            robot_type,
            EmbodimentTag.NEW_EMBODIMENT,
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    if hasattr(data_config, "make_dataset"):
        return data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=(data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"),
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
        )

    return BehaviorSkillSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=(data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"),
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> BehaviorSkillMixtureDataset:
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info("[behavior-skill] Using mixture '%s': %s", data_mix, [(d, w, r) for d, w, r in mixture_spec])

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            logger.info("Skipping duplicate dataset: %r", (d_name, d_weight, robot_type))
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_behavior_skill_dataset(
                    Path(data_root_dir),
                    d_name,
                    robot_type,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                ),
                d_weight,
            )
        )

    return BehaviorSkillMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


def make_behavior_skill_sampler(dataset: BehaviorSkillMixtureDataset) -> BehaviorSkillEpochSampler:
    return BehaviorSkillEpochSampler(dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/Behavior-Skill/train_files/SV001_starvla_behavior_skill_easy_qwen35_08b.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override data_root_dir from config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = Path(vla_dataset_cfg.data_root_dir)
    if args.data_mix is not None:
        vla_dataset_cfg.data_mix = args.data_mix
    if args.data_root_dir is not None:
        vla_dataset_cfg.data_root_dir = Path(args.data_root_dir)

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 3:
            break
        count += 1
