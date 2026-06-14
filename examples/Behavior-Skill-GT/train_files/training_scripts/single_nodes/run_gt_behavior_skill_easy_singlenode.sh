#!/usr/bin/env bash
# ============================================================================
# 单机多卡(默认 8 卡)训练入口。两个配置都用这一个脚本启动:
#   默认(subtask_uniform):
#     bash examples/Behavior-Skill-GT/train_files/training_scripts/single_nodes/run_gt_behavior_skill_easy_singlenode.sh
#   切 size_proportional:
#     CONFIG_YAML=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_easy_qwen35_08b_size_proportional.yaml \
#       bash .../single_nodes/run_gt_behavior_skill_easy_singlenode.sh
# 常用覆盖: NUM_PROCESSES=8 / BEHAVIOR_SKILL_DATA_ROOT=... / RUN_ID=... / RUN_ROOT_DIR=...
# ============================================================================
# ---- 单机网络/NCCL(loopback, IB 关闭)----
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# ---- 共用主体: cd repo 根 + 解析配置 + 组装 LAUNCH_TAIL[] ----
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_common.sh"

num_processes="${NUM_PROCESSES:-8}"

echo "[singlenode] CONFIG_YAML=${config_yaml}"
echo "[singlenode] num_processes=${num_processes}  data_root=${behavior_skill_data_root}"
echo "[singlenode] output_dir=${output_dir}"

accelerate launch \
  --config_file "${accel_ds_config}" \
  --num_machines 1 \
  --num_processes "${num_processes}" \
  "${LAUNCH_TAIL[@]}"
