import numpy as np


# Official BEHAVIOR-1K / OpenPI action-state layout for R1Pro:
# [base_qvel(3), torso_qpos(4), left_arm_qpos(7), left_gripper_width(1),
#  right_arm_qpos(7), right_gripper_width(1)]
B1K_ACTION23_SLICES = {
    "action.base": slice(0, 3),
    "action.torso": slice(3, 7),
    "action.left_arm": slice(7, 14),
    "action.left_gripper": slice(14, 15),
    "action.right_arm": slice(15, 22),
    "action.right_gripper": slice(22, 23),
}

B1K_ORDERED_ACTION_KEYS = list(B1K_ACTION23_SLICES.keys())

# OpenPI DeltaActions(make_bool_mask(-3, 3, -1, 7, -1, 7, -1))
B1K_DELTA_PREFIX_DIMS = {
    "action.torso": 3,
    "action.left_arm": 7,
    "action.right_arm": 7,
}

R1PRO_PROPRIO_SLICES = {
    "arm_left_qpos": slice(158, 165),
    "gripper_left_qpos": slice(193, 195),
    "arm_right_qpos": slice(197, 204),
    "gripper_right_qpos": slice(232, 234),
    "trunk_qpos": slice(236, 240),
    "base_qvel": slice(253, 256),
}

B1K_MAX_GRIPPER_WIDTH = 0.1


def get_b1k_delta_action_slice(action_key: str) -> slice | None:
    action_slice = B1K_ACTION23_SLICES.get(action_key)
    if action_slice is None:
        return None
    delta_dim = B1K_DELTA_PREFIX_DIMS.get(action_key, 0)
    if delta_dim <= 0:
        return None
    return slice(action_slice.start, action_slice.start + delta_dim)


def extract_b1k_state_from_proprio(proprio_data: np.ndarray) -> np.ndarray:
    """Match the official behavior-1k state layout used by R1Pro."""
    proprio_data = np.asarray(proprio_data, dtype=np.float32)

    base_qvel = proprio_data[..., R1PRO_PROPRIO_SLICES["base_qvel"]]
    trunk_qpos = proprio_data[..., R1PRO_PROPRIO_SLICES["trunk_qpos"]]
    arm_left_qpos = proprio_data[..., R1PRO_PROPRIO_SLICES["arm_left_qpos"]]
    arm_right_qpos = proprio_data[..., R1PRO_PROPRIO_SLICES["arm_right_qpos"]]

    left_gripper_raw = proprio_data[..., R1PRO_PROPRIO_SLICES["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., R1PRO_PROPRIO_SLICES["gripper_right_qpos"]].sum(axis=-1, keepdims=True)

    left_gripper_width = 2.0 * (left_gripper_raw / B1K_MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / B1K_MAX_GRIPPER_WIDTH) - 1.0

    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            left_gripper_width,
            arm_right_qpos,
            right_gripper_width,
        ],
        axis=-1,
    ).astype(np.float32)
