import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch.distributed as dist

from starVLA.dataloader.gr00t_lerobot.behavior1k_utils import (
    B1K_ACTION23_SLICES,
    B1K_DELTA_PREFIX_DIMS,
    B1K_ORDERED_ACTION_KEYS,
    extract_b1k_state_from_proprio,
    get_b1k_delta_action_slice,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LE_ROBOT_DATA_FILENAME,
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_MODALITY_FILENAME,
    LE_ROBOT_STATS_FILENAME,
    LE_ROBOT3_EPISODE_FILENAME,
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotSingleDataset,
    LeRobotStateActionMetadata,
    _normalize_action_mode,
    _normalize_action_mode_apply_keys,
    _normalize_action_mode_state_map,
)
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


# Behavior_Skill_V1.0 已知坏 parquet(128KB 截断, 14万段里唯一一个):
#   hard/wipe/wipe_the_trumpet/data/episode_03700117.parquet
# 按 (subtask 目录名, episode_index) 匹配, 在 get_trajectory_ids 里跳过该 ep ->
# 它永不进采样池、永不被 __getitem__ 读到; 该段其余 199 个 ep 照常训练。
_BAD_EPISODES = {("wipe_the_trumpet", 3700117)}


def _is_bad_episode(dataset_path: Path, episode_index) -> bool:
    try:
        return (Path(dataset_path).name, int(episode_index)) in _BAD_EPISODES
    except (TypeError, ValueError):
        return False


def _resolve_repo_relative_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[3]
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return repo_root / path


def _load_external_norm_stats(path_str: str) -> dict:
    stats_path = _resolve_repo_relative_path(path_str)
    if not stats_path.exists():
        raise FileNotFoundError(f"External norm stats file not found: {stats_path}")
    print(f"Loaded external action norm stats from {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "norm_stats" in payload and isinstance(payload["norm_stats"], dict):
        payload = payload["norm_stats"]
    return payload


def _slice_external_stats_array(value: list | np.ndarray, start: int, end: int) -> list:
    array = np.asarray(value)
    if array.ndim == 1:
        if end > array.shape[0]:
            raise ValueError(f"External stats dim {array.shape[0]} is smaller than requested slice [{start}:{end}]")
        return array[start:end].tolist()
    if array.ndim == 2:
        if end > array.shape[1]:
            raise ValueError(
                f"External per-timestamp stats dim {array.shape[1]} is smaller than requested slice [{start}:{end}]"
            )
        return array[:, start:end].tolist()
    raise ValueError(f"Unsupported external stats shape: {array.shape}")


def _build_action_statistics_from_external_norm_stats(
    external_norm_stats: dict,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
) -> dict[str, dict]:
    action_block = external_norm_stats.get("actions") or external_norm_stats.get("action")
    if not isinstance(action_block, dict):
        raise ValueError("External norm stats must contain an `actions` or `action` block.")

    required_stats = ["mean", "std", "q01", "q99"]
    fallback_stats = {"min": "q01", "max": "q99"}
    optional_stats = ["per_timestamp_mean", "per_timestamp_std", "per_timestamp_q01", "per_timestamp_q99"]
    action_statistics: dict[str, dict] = {}

    global_offsets = {}
    cursor = 0
    for action_key in B1K_ORDERED_ACTION_KEYS:
        action_slice = B1K_ACTION23_SLICES[action_key]
        width = action_slice.stop - action_slice.start
        global_offsets[action_key] = (cursor, cursor + width)
        cursor += width

    for action_key in action_keys_full:
        action_subkey = action_key.replace("action.", "", 1)
        if action_subkey not in lerobot_modality_meta.action:
            raise ValueError(f"Action key missing in metadata: {action_key}")
        if action_key not in global_offsets:
            raise ValueError(f"Unsupported action key for external Behavior stats: {action_key}")

        start, end = global_offsets[action_key]
        sliced_stats = {}
        for stat_name in required_stats:
            if stat_name not in action_block:
                raise ValueError(f"Missing required statistic `{stat_name}` in external norm stats.")
            sliced_stats[stat_name] = _slice_external_stats_array(action_block[stat_name], start, end)
        for stat_name, fallback_name in fallback_stats.items():
            stat_value = action_block.get(stat_name, action_block.get(fallback_name))
            if stat_value is None:
                raise ValueError(
                    f"Missing required statistic `{stat_name}` in external norm stats and no `{fallback_name}` fallback was provided."
                )
            sliced_stats[stat_name] = _slice_external_stats_array(stat_value, start, end)
        for stat_name in optional_stats:
            stat_value = action_block.get(stat_name)
            sliced_stats[stat_name] = (
                _slice_external_stats_array(stat_value, start, end) if stat_value is not None else None
            )
        action_statistics[action_subkey] = sliced_stats

    return action_statistics


def _matches_source_span_name(
    source_span_name: str | None,
    allowed_source_span_names: set[str] | None,
) -> bool:
    if allowed_source_span_names is None:
        return True
    source_span_name = "" if source_span_name is None else str(source_span_name)
    return any(pattern in source_span_name for pattern in allowed_source_span_names)


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


def _build_b1k_action_ref_config(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    action_mode_apply_keys: list[str] | None,
) -> dict[str, list[tuple[tuple[int, int], tuple[int, int], str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_meta = lerobot_modality_meta.action
    action_col_slices: dict[str, list[tuple[tuple[int, int], tuple[int, int], str]]] = {}
    for action_key in apply_keys:
        action_subkey = action_key.replace("action.", "", 1)
        if action_subkey not in action_meta:
            raise ValueError(f"Action key missing in metadata: {action_key}")
        action_cfg = action_meta[action_subkey]
        action_col = action_cfg.original_key or action_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_ref_slice = get_b1k_delta_action_slice(action_key)
        if state_ref_slice is None:
            continue
        action_dim = action_slice[1] - action_slice[0]
        qpos_dim = B1K_ACTION23_SLICES[action_key].stop - B1K_ACTION23_SLICES[action_key].start
        delta_dim = _get_b1k_delta_prefix_dim(action_key, action_dim, qpos_dim)
        if delta_dim <= 0:
            continue
        action_padding = "first_last" if action_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            ((action_slice[0], action_slice[0] + delta_dim), (state_ref_slice.start, state_ref_slice.stop), action_padding)
        )
    return action_col_slices


class BehaviorSkillSingleDataset(LeRobotSingleDataset):
    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel", "b1k_delta"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel, b1k_delta.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _get_metadata(self, embodiment_tag) -> DatasetMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                continuous = bool(np.issubdtype(state_action_dtype, np.floating))
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": continuous,
                }

        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key or new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                try:
                    channels = le_video_meta["info"]["video.channels"]
                    fps = le_video_meta["info"]["video.fps"]
                except (ValueError, KeyError):
                    channels = 3
                    fps = le_info.get("fps", 30)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")
        action_cfg = self.modality_configs.get("action")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        external_action_norm_stats_path = self.data_cfg.get("action_norm_stats_path", None) if self.data_cfg else None
        if not external_action_norm_stats_path:
            raise ValueError(
                "Behavior-Skill requires datasets.vla_data.action_norm_stats_path. "
                "Please provide the 50tasks norm_stats.json path."
            )

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        include_source_span_names = self.data_cfg.get("include_source_span_names") if self.data_cfg else None
        if isinstance(include_source_span_names, str):
            include_source_span_names = [include_source_span_names]
        external_norm_stats = _load_external_norm_stats(str(external_action_norm_stats_path))
        dataset_statistics = {
            "state": {},
            "action": _build_action_statistics_from_external_norm_stats(
                external_norm_stats=external_norm_stats,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
            ),
        }

        for stat in dataset_statistics["action"].values():
            DatasetStatisticalValues.model_validate(stat)

        return DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        include_source_span_names = None
        if self.data_cfg is not None:
            include_source_span_names = self.data_cfg.get("include_source_span_names")
        if isinstance(include_source_span_names, str):
            include_source_span_names = [include_source_span_names]
        allowed_source_span_names = (
            {str(name) for name in include_source_span_names}
            if include_source_span_names
            else None
        )

        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                if (
                    allowed_source_span_names is not None
                    and not _matches_source_span_name(
                        episode.get("source_span_name"), allowed_source_span_names
                    )
                ):
                    continue
                if _is_bad_episode(self.dataset_path, episode["episode_index"]):
                    continue   # 跳过已知坏 parquet 对应的 ep(见 _BAD_EPISODES)
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            if allowed_source_span_names is not None:
                print(
                    f"Filtered {self.dataset_name} to source_span_name containing "
                    f"{sorted(allowed_source_span_names)}: {len(trajectory_ids)} episodes kept"
                )
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        if self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    if _is_bad_episode(self.dataset_path, episode["episode_index"]):
                        continue   # 防御性: v3.0 路径同样跳过坏 ep(见 _BAD_EPISODES)
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])
                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)
                    video_file_indices = {}
                    for col in timestamp_cols:
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        if chunk_col in episode and file_col in episode:
                            video_file_indices[video_key] = {
                                "chunk_index": int(episode[chunk_col]),
                                "file_index": int(episode[file_col]),
                            }
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                        "videos/file_indices": video_file_indices,
                    }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        raise ValueError(f"Unsupported lerobot version: {self._lerobot_version}")

    def _get_steps_config_key(self) -> str:
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
            "include_source_span_names": self.data_cfg.get("include_source_span_names") if self.data_cfg is not None else None,
        }
        config_str = str(sorted(config_dict.items()))
        return __import__("hashlib").md5(config_str.encode()).hexdigest()[:12]

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        if self._action_mode == "b1k_delta":
            raw_proprio_key = "__b1k_observation_state"
            if raw_proprio_key not in data:
                raise ValueError(
                    f"B1K delta requires {raw_proprio_key} in sample data. "
                    f"Available keys: {sorted(data.keys())}"
                )
            official_state23 = extract_b1k_state_from_proprio(np.asarray(data[raw_proprio_key])[0])
            action_keys = self._action_mode_apply_keys or []
            for action_key in action_keys:
                if action_key not in data:
                    continue
                action_values = np.asarray(data[action_key])
                if action_values.ndim != 2:
                    raise ValueError(f"Expected 2D array for {action_key}, got {action_values.shape}")
                action_slice = B1K_ACTION23_SLICES.get(action_key)
                if action_slice is None:
                    continue
                state_ref_slice = get_b1k_delta_action_slice(action_key)
                delta_dim = _get_b1k_delta_prefix_dim(
                    action_key,
                    action_values.shape[1],
                    action_slice.stop - action_slice.start,
                )
                if delta_dim <= 0:
                    continue
                out = action_values.copy()
                out[:, :delta_dim] = action_values[:, :delta_dim] - official_state23[state_ref_slice]
                data[action_key] = out
            data.pop(raw_proprio_key, None)
            return data

        return super()._apply_action_mode(data)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        if self._action_mode == "b1k_delta":
            assert self.curr_traj_data is not None
            if "observation.state" not in self.curr_traj_data.columns:
                raise ValueError(
                    "B1K delta requires raw proprio column observation.state "
                    "to reproduce behavior-1k state extraction."
                )
            data["__b1k_observation_state"] = np.asarray(
                [self.curr_traj_data["observation.state"].iloc[base_index]],
                dtype=np.float32,
            )
        data = self._apply_action_mode(data)
        return data

    # ---- 源视频直读(可选): datasets.vla_data.source_video_root 指向原始 behavior-1k_demo 的 videos/ 根 ----
    # 开启后跳过切段 mp4, 直接按 parquet 的 source_episode_index/source_timestamp 去原视频取帧
    # (无损像素; 路径规律 task-{src//10^4:04d}/{video_key}/episode_{src:08d}.mp4, 与切段血缘已逐位验证一致)。
    # 开关 datasets.vla_data.use_local_videos: true 时强制读各 subtask 自己 videos/ 下的切段(忽略 source_video_root)。
    # 两者都不配置则行为与基类完全相同。

    def _get_source_video_root(self) -> Path | None:
        if not hasattr(self, "_source_video_root"):
            use_local = bool(self.data_cfg.get("use_local_videos", False)) if self.data_cfg else False
            root = None if use_local else (self.data_cfg.get("source_video_root", None) if self.data_cfg else None)
            self._source_video_root = Path(root) if root else None
        return self._source_video_root

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        source_root = self._get_source_video_root()
        if source_root is None:
            return super().get_video(trajectory_id, key, base_index)

        # 帧窗口与越界 padding 逻辑与基类 get_video 保持一致
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        key = key.replace("video.", "")
        original_key = self.lerobot_modality_meta.video[key].original_key or key

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        for col in ("source_episode_index", "source_timestamp"):
            if col not in self.curr_traj_data.columns:
                raise ValueError(
                    f"source_video_root requires parquet column '{col}' (missing in {trajectory_id=})"
                )
        source_episode = int(self.curr_traj_data["source_episode_index"].iloc[0])
        # source_timestamp 即原视频时间轴上的秒数(= source_frame_index / fps)
        video_timestamp = self.curr_traj_data["source_timestamp"].to_numpy()[step_indices]
        video_path = (
            source_root
            / f"task-{source_episode // 10000:04d}"
            / original_key
            / f"episode_{source_episode:08d}.mp4"
        )
        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
