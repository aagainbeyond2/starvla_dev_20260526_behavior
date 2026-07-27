"""Helpers for exact, node-local training-state checkpoints.

The H100 training nodes use independent local filesystems. During a
DeepSpeed ZeRO-2 save, every rank writes a different optimizer shard, so a
checkpoint is only loadable after the per-node directories have been merged.
This module keeps checkpoint discovery, metadata, sampler positioning, and
local retention independent from the trainer implementation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

FULL_STATE_FORMAT_VERSION = 1
FULL_STATE_DIRNAME = "full_state_checkpoints"
LOCAL_COMPLETE_MARKER = "_LOCAL_COMPLETE"
MERGED_COMPLETE_MARKER = "_MERGED_COMPLETE"
METADATA_FILENAME = "trainer_state.json"

_STEP_DIR_RE = re.compile(r"^steps_(\d+)$")


@dataclass(frozen=True)
class FullStateMetadata:
    """Trainer-owned state not covered by DeepSpeed / Accelerate."""

    completed_steps: int
    consumed_batches: int
    data_epoch: int
    batches_into_epoch: int
    world_size: int
    local_world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    eval_interval: int
    format_version: int = FULL_STATE_FORMAT_VERSION

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FullStateMetadata":
        metadata = cls(**{field: int(value) for field, value in payload.items()})
        if metadata.format_version != FULL_STATE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported full-state format {metadata.format_version}; "
                f"expected {FULL_STATE_FORMAT_VERSION}."
            )
        if min(
            metadata.completed_steps,
            metadata.consumed_batches,
            metadata.data_epoch,
            metadata.batches_into_epoch,
        ) < 0:
            raise ValueError(f"Negative counters in full-state metadata: {payload}")
        if min(
            metadata.world_size,
            metadata.local_world_size,
            metadata.per_device_batch_size,
            metadata.gradient_accumulation_steps,
            metadata.eval_interval,
        ) <= 0:
            raise ValueError(f"Non-positive topology/config values in full-state metadata: {payload}")
        return metadata


def validate_resume_compatibility(
    metadata: FullStateMetadata,
    *,
    world_size: int,
    local_world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    eval_interval: int,
) -> None:
    """Reject topology/config changes that would alter exact continuation semantics."""

    expected = {
        "world_size": int(world_size),
        "local_world_size": int(local_world_size),
        "per_device_batch_size": int(per_device_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "eval_interval": int(eval_interval),
    }
    mismatches = {
        key: {"checkpoint": int(getattr(metadata, key)), "current": current}
        for key, current in expected.items()
        if int(getattr(metadata, key)) != current
    }
    if mismatches:
        raise RuntimeError(
            "Full-state checkpoint is incompatible with the current exact-resume "
            f"configuration: {mismatches}"
        )


def atomic_write_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Write JSON in the destination directory and publish it atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def full_state_root(output_dir: Path | str) -> Path:
    return Path(output_dir) / FULL_STATE_DIRNAME


def full_state_step_dir(root: Path | str, step: int) -> Path:
    return Path(root) / f"steps_{int(step):09d}"


def full_state_incomplete_dir(root: Path | str, step: int) -> Path:
    return Path(root) / f".steps_{int(step):09d}.incomplete"


def load_full_state_metadata(step_dir: Path | str) -> FullStateMetadata:
    step_dir = Path(step_dir)
    match = _STEP_DIR_RE.match(step_dir.name)
    if match is None:
        raise ValueError(f"Invalid full-state checkpoint directory name: {step_dir}")
    with open(step_dir / METADATA_FILENAME, "r", encoding="utf-8") as f:
        metadata = FullStateMetadata.from_dict(json.load(f))
    dirname_step = int(match.group(1))
    if metadata.completed_steps != dirname_step:
        raise ValueError(
            f"Checkpoint directory step {dirname_step} disagrees with metadata "
            f"step {metadata.completed_steps}: {step_dir}"
        )
    return metadata


def iter_complete_full_state_dirs(
    root: Path | str,
    *,
    require_merged: bool,
) -> Iterable[tuple[int, Path]]:
    root = Path(root)
    if not root.is_dir():
        return []

    completed: list[tuple[int, Path]] = []
    marker = MERGED_COMPLETE_MARKER if require_merged else LOCAL_COMPLETE_MARKER
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _STEP_DIR_RE.match(path.name)
        if match is None:
            continue
        if not (path / marker).is_file() or not (path / METADATA_FILENAME).is_file():
            continue
        try:
            metadata = load_full_state_metadata(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        completed.append((metadata.completed_steps, path))
    completed.sort(key=lambda item: item[0])
    return completed


def latest_full_state_dir(
    root: Path | str,
    *,
    require_merged: bool,
) -> tuple[Path | None, FullStateMetadata | None]:
    completed = list(iter_complete_full_state_dirs(root, require_merged=require_merged))
    if not completed:
        return None, None
    _, path = completed[-1]
    return path, load_full_state_metadata(path)


def prune_local_full_states(root: Path | str, keep_last: int) -> list[Path]:
    """Remove older locally-complete full states, retaining at least one."""

    keep_last = max(1, int(keep_last))
    completed = list(iter_complete_full_state_dirs(root, require_merged=False))
    removed: list[Path] = []
    for _, path in completed[:-keep_last]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def find_resume_sampler(dataloader: Any) -> Any | None:
    """Find the custom epoch sampler through Accelerate's wrapper layers."""

    queue = [dataloader]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if callable(getattr(current, "set_resume_sample_offset", None)):
            return current
        for attribute in ("sampler", "batch_sampler", "dataloader", "base_dataloader"):
            child = getattr(current, attribute, None)
            if child is not None:
                queue.append(child)
    return None


def restore_dataloader_position(
    dataloader: Any,
    metadata: FullStateMetadata,
    *,
    split_batches: bool,
) -> int:
    """Restore epoch and cursor without decoding/skipping prior video batches.

    With Accelerate's default ``split_batches=False``, every process consumes
    one distinct per-device batch from a global run of ``world_size`` batches.
    Therefore ``batches_into_epoch`` per rank maps to the same global sampler
    offset on every rank.
    """

    if split_batches:
        raise RuntimeError("Exact sampler resume currently requires Accelerator split_batches=False.")
    sampler = find_resume_sampler(dataloader)
    if sampler is None:
        raise RuntimeError(
            "Could not find a sampler with set_resume_sample_offset(); "
            "full-state resume cannot restore the data cursor exactly."
        )

    if callable(getattr(dataloader, "set_epoch", None)):
        dataloader.set_epoch(metadata.data_epoch)
    else:
        sampler.set_epoch(metadata.data_epoch)
    sample_offset = (
        metadata.batches_into_epoch
        * metadata.world_size
        * metadata.per_device_batch_size
    )
    sampler.set_resume_sample_offset(sample_offset)
    return sample_offset
