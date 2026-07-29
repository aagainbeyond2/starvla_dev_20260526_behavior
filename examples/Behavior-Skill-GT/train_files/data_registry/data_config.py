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


# GT 混采：data_root_dir = Behavior_Skill_V1.0/easy(与 v4/easy 叶子名/episode数完全一致, 可互换)，子集名为两级 skill_type/skill_subtask
# (loader 用 dataset_path = data_root_dir / name)。全量 34 个 skill_subtask / 10588 eps。
DATASET_NAMED_MIXTURES = {
    # 冒烟用：单 subtask
    "gt_behavior_skill_smoke": [
        ("close_door/close_the_fridge_door", 1.0, "R1ProSkill"),
    ],
    # Behavior-1K 2025 challenge Easy4 原始四任务 / 800 episodes。
    # 配置中的 data_root_dir 指 datasets_training/training_data 父目录。
    "gt_behavior_easy4": [
        ("behavior-1k-easy4", 1.0, "R1ProSkill"),
    ],
    # 全量 easy：34 个 skill_subtask
    "gt_behavior_skill_easy": [
        ("close_door/close_the_fridge_door", 1.0, "R1ProSkill"),
        ("close_door/close_the_microwave_door", 1.0, "R1ProSkill"),
        ("close_door/close_the_washer_door", 1.0, "R1ProSkill"),
        ("open_door/open_the_door", 1.0, "R1ProSkill"),
        ("open_door/open_the_fridge_door", 1.0, "R1ProSkill"),
        ("open_door/open_the_microwave_door", 1.0, "R1ProSkill"),
        ("open_door/open_the_washer_door", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_baseball_cap_from_the_countertop", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_beer_bottle_from_the_fridge", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_can_of_soda_from_the_floors", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_frying_pan_from_the_burner", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_hotdog_from_the_countertop", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_hotdog_from_the_fridge", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_popcorn_bag_from_the_bar", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_radio_from_the_coffee_table", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_storage_box_from_the_floors", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_trash_can_from_the_floors", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_tray_from_the_countertop", 1.0, "R1ProSkill"),
        ("pick_up/pick_up_the_tray_from_the_fridge", 1.0, "R1ProSkill"),
        ("place/place_the_baseball_cap_in_the_washer", 1.0, "R1ProSkill"),
        ("place/place_the_beer_bottle_on_the_coffee_table", 1.0, "R1ProSkill"),
        ("place/place_the_can_of_soda_in_the_trash_can", 1.0, "R1ProSkill"),
        ("place/place_the_frying_pan_on_the_burner", 1.0, "R1ProSkill"),
        ("place/place_the_hotdog_in_the_microwave", 1.0, "R1ProSkill"),
        ("place/place_the_hotdog_on_the_countertop", 1.0, "R1ProSkill"),
        ("place/place_the_popcorn_bag_in_the_microwave", 1.0, "R1ProSkill"),
        ("place/place_the_storage_box_on_the_floors", 1.0, "R1ProSkill"),
        ("place/place_the_storage_box_on_the_storage_box", 1.0, "R1ProSkill"),
        ("place/place_the_tray_on_the_countertop_next_to_the_burner", 1.0, "R1ProSkill"),
        ("pour/pour_the_bacons_from_the_tray_into_the_frying_pan", 1.0, "R1ProSkill"),
        ("press/press_the_radio", 1.0, "R1ProSkill"),
        ("turn_on_switch/turn_on_the_burner_switch", 1.0, "R1ProSkill"),
        ("turn_on_switch/turn_on_the_microwave_switch", 1.0, "R1ProSkill"),
        ("turn_on_switch/turn_on_the_washer_switch", 1.0, "R1ProSkill"),
    ],
}

# 全量混采 gt_behavior_skill_full(471 subtask = easy 34 + normal 169 + hard 268, ~14万 ep)。
# 由 gen_full_mixture.py 扫目录生成的 gt_behavior_skill_full.json 注册(避免 471 行硬编码;改数据重跑 gen)。
# name 三级 <difficulty>/<skill_type>/<skill_subtask>, 配置里 data_root_dir 指 Behavior_Skill_V1.0 父目录。
# 坏 episode(wipe_the_trumpet/03700117)整段保留, 坏帧在 loader(_BAD_EPISODES)按 episode_index 跳过。
import json as _json, os as _os  # noqa: E402
_full_json = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "gt_behavior_skill_full.json")
if _os.path.exists(_full_json):
    DATASET_NAMED_MIXTURES["gt_behavior_skill_full"] = [
        tuple(x) for x in _json.load(open(_full_json, encoding="utf-8"))
    ]
