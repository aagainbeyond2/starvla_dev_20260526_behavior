import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from .state_action import RotationTransform, RotationType, StateActionMetadata, StateActionTransform


class BehaviorQ99PerTimeTransform(StateActionTransform):
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )

    _q01_tensors: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _q99_tensors: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _pt_q01_tensors: dict[str, torch.Tensor | None] = PrivateAttr(default_factory=dict)
    _pt_q99_tensors: dict[str, torch.Tensor | None] = PrivateAttr(default_factory=dict)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(config, StateActionMetadata), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key not in self.normalization_modes:
                continue
            normalization_mode = self.normalization_modes[modality_key]
            if normalization_mode != "q99":
                raise ValueError(
                    "BehaviorQ99PerTimeTransform only supports q99 normalization, "
                    f"but got {normalization_mode!r} for {modality_key}."
                )
            assert "q01" in normalization_statistics and "q99" in normalization_statistics, (
                f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
            )
            assert len(normalization_statistics["q01"]) == len(normalization_statistics["q99"]), (
                "q01 and q99 statistics must have the same length, "
                f"but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
            )
        return self

    def set_metadata(self, dataset_metadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(modality_metadata, modality), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        for key in self.normalization_modes:
            modality, state_key = key.split(".", 1)
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(dataset_statistics, modality), f"{state_key} statistics not found"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[state_key].model_dump()

        for key in self.target_rotations:
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"
            to_rep = RotationType(self.target_rotations[key])
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(from_rep=from_rep.value, to_rep=to_rep.value)

        for key in self.normalization_modes:
            if key in self._rotation_transformers:
                raise ValueError("BehaviorQ99PerTimeTransform does not support rotation-converted q99 normalization.")
            stats = self.normalization_statistics[key]
            self._q01_tensors[key] = torch.tensor(stats["q01"])
            self._q99_tensors[key] = torch.tensor(stats["q99"])
            self._pt_q01_tensors[key] = (
                torch.tensor(stats["per_timestamp_q01"]) if stats.get("per_timestamp_q01") is not None else None
            )
            self._pt_q99_tensors[key] = (
                torch.tensor(stats["per_timestamp_q99"]) if stats.get("per_timestamp_q99") is not None else None
            )

    def _get_q99_pair(self, key: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pt_q01 = self._pt_q01_tensors.get(key)
        pt_q99 = self._pt_q99_tensors.get(key)
        # Per-timestamp q01/q99 are mandatory here: evaluation-time unnormalization
        # (serve_starvla_for_eval._unnormalize_behavior_q99_actions) always uses the
        # per_timestamp stats, so silently falling back to the global q01/q99 would
        # make training and inference diverge. Fail loudly instead.
        if pt_q01 is None or pt_q99 is None:
            raise ValueError(
                f"BehaviorQ99PerTimeTransform requires per-timestamp q01/q99 statistics for {key!r}, "
                "but they are missing from the norm stats. Provide `per_timestamp_q01`/`per_timestamp_q99` "
                "in the external action norm stats to stay consistent with eval-time unnormalization."
            )
        if x.ndim < 2:
            raise ValueError(
                f"BehaviorQ99PerTimeTransform expects at least 2D tensors (time, dim) for {key!r}, "
                f"got shape {tuple(x.shape)}."
            )
        return (
            pt_q01[: x.shape[-2], : x.shape[-1]].to(x.dtype),
            pt_q99[: x.shape[-2], : x.shape[-1]].to(x.dtype),
        )

    def apply(self, data: dict):
        for key in self.apply_to:
            if key not in data:
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(input_dtype, torch.dtype), (
                    f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                )
                self._input_dtypes[key] = input_dtype
            else:
                assert data[key].dtype == self._input_dtypes[key], (
                    "All states corresponding to the same key must be of the same dtype, "
                    f"input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
                )
            state = data[key]
            q01, q99 = self._get_q99_pair(key, state)
            normalized = torch.clamp((state - q01) / (q99 - q01 + 1e-6) * 2 - 1, -1, 1)
            data[key] = normalized
        return data

    def unapply(self, data: dict):
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(state, torch.Tensor), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            q01, q99 = self._get_q99_pair(key, state)
            state = (state + 1) / 2 * (q99 - q01 + 1e-6) + q01
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data
