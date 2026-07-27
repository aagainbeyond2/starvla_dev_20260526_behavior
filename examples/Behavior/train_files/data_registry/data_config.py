"""Data registration for long-horizon BEHAVIOR tasks."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class R1ProDataConfig:
    """R1 Pro observations/actions used by the original BEHAVIOR trajectories."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.head",
        "video.left_wrist",
        "video.right_wrist",
    ]
    state_keys = [
        "state.joint_qpos_sin",
        "state.joint_qpos_cos",
    ]
    action_keys = [
        "action.base",
        "action.torso",
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(30))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "R1Pro": R1ProDataConfig(),
}


DATASET_NAMED_MIXTURES = {
    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],
    "behavior_easy4": [
        ("behavior-1k-easy4", 1.0, "R1Pro"),
    ],
}
