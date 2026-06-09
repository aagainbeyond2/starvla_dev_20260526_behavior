from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.behavior_skill_dataset import BehaviorSkillSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.behavior_state_action import BehaviorQ99PerTimeTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoRandomGrayscale,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class R1ProSkillDataConfig:
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
    action_dim_names = [
        "base.vx",
        "base.vy",
        "base.wz",
        "torso.j0",
        "torso.j1",
        "torso.j2",
        "torso.j3",
        "left_arm.j0",
        "left_arm.j1",
        "left_arm.j2",
        "left_arm.j3",
        "left_arm.j4",
        "left_arm.j5",
        "left_arm.j6",
        "left_gripper",
        "right_arm.j0",
        "right_arm.j1",
        "right_arm.j2",
        "right_arm.j3",
        "right_arm.j4",
        "right_arm.j5",
        "right_arm.j6",
        "right_gripper",
    ]
    action_dim_groups = {
        "base": (0, 3),
        "torso": (3, 7),
        "left_arm": (7, 14),
        "left_gripper": (14, 15),
        "right_arm": (15, 22),
        "right_gripper": (22, 23),
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(30))
    video_native_resolutions = {
        "video.head": (720, 720),
        "video.left_wrist": (480, 480),
        "video.right_wrist": (480, 480),
    }

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def get_action_dim_names(self):
        return list(self.action_dim_names)

    def get_action_dim_groups(self):
        return dict(self.action_dim_groups)

    def _video_augmentation_transforms(self, data_cfg=None):
        image_aug_cfg = self._cfg_get(data_cfg, "image_augmentation", None)
        if not self._cfg_get(image_aug_cfg, "enabled", False):
            return []

        image_resolution = self._cfg_get(data_cfg, "default_image_resolution", [3, 224, 224])
        target_height = int(image_resolution[1]) if len(image_resolution) >= 2 else 224
        target_width = int(image_resolution[2]) if len(image_resolution) >= 3 else 224

        transforms = []
        # Behavior-Skill mixes 720x720 and 480x480 camera streams, so convert/resize each
        # view independently before applying shared multi-view augmentations.
        for video_key in self.video_keys:
            transforms.append(VideoToTensor(apply_to=[video_key]))
        for video_key in self.video_keys:
            transforms.append(
                VideoResize(
                    apply_to=[video_key],
                    height=target_height,
                    width=target_width,
                    interpolation="linear",
                )
            )

        brightness = float(self._cfg_get(image_aug_cfg, "brightness", 0.0))
        contrast = float(self._cfg_get(image_aug_cfg, "contrast", 0.0))
        saturation = float(self._cfg_get(image_aug_cfg, "saturation", 0.0))
        hue = float(self._cfg_get(image_aug_cfg, "hue", 0.0))
        if any(value > 0.0 for value in (brightness, contrast, saturation, hue)):
            transforms.append(
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

        random_grayscale_p = float(self._cfg_get(image_aug_cfg, "random_grayscale_p", 0.0))
        if random_grayscale_p > 0.0:
            transforms.append(VideoRandomGrayscale(apply_to=self.video_keys, p=random_grayscale_p))

        transforms.append(VideoToNumpy(apply_to=self.video_keys))
        return transforms

    def modality_config(self, data_cfg=None):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, data_cfg=None):
        return ComposedModalityTransform(
            transforms=[
                *self._video_augmentation_transforms(data_cfg=data_cfg),
                StateActionToTensor(apply_to=self.action_keys),
                BehaviorQ99PerTimeTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )

    def make_dataset(self, **dataset_kwargs):
        return BehaviorSkillSingleDataset(**dataset_kwargs)


ROBOT_TYPE_CONFIG_MAP = {
    "R1ProSkill": R1ProSkillDataConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "R1ProSkill": EmbodimentTag.NEW_EMBODIMENT,
}


DATASET_NAMED_MIXTURES = {
    "behavior_skill_debug": [
        ("task-0000", 1.0, "R1ProSkill"),
    ],
    "behavior_skill_easy": [
        ("task-0000", 1.0, "R1ProSkill"),
        ("task-0001", 1.0, "R1ProSkill"),
        ("task-0016", 1.0, "R1ProSkill"),
        ("task-0017", 1.0, "R1ProSkill"),
        ("task-0032", 1.0, "R1ProSkill"),
        ("task-0040", 1.0, "R1ProSkill"),
        ("task-0045", 1.0, "R1ProSkill"),
        ("task-0046", 1.0, "R1ProSkill"),
    ],
}
