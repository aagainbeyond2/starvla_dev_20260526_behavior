#!/usr/bin/env bash
# ============================================================================
# 写死配置的单机训练启动器 —— 【0.8B subtask_uniform(基线)】
# 直接 bash 本脚本即可, 无需任何参数(config 已写死, 避免传参被无视的坑)。
#   bash examples/Behavior-Skill-GT/train_files/training_scripts/train_08b_uniform.sh
# 自包含: 自己 unset pod 多机 env + 激活 venv + cd 子模块根 + 用 CONFIG_YAML 传给内层入口。
# 不依赖 train_singlenode.sh。建议在 tmux 里跑: tmux new -s wgt; Ctrl-b d 脱离。
# ============================================================================
CONFIG="examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_easy_qwen35_08b_subtask_uniform.yaml"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
ENTRY="examples/Behavior-Skill-GT/train_files/training_scripts/single_nodes/run_gt_behavior_skill_easy_singlenode.sh"
VENV=/opt/venv/starvla-qwen35

# 1) 清掉 pod 注入的多机分布式变量(否则非 master 节点拿 rank 8-15, dist.barrier() 卡死)
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK NNODES NPROC_PER_NODE \
      MASTER_ADDR MASTER_PORT PET_NNODES PET_NODE_RANK PET_NPROC_PER_NODE PET_MASTER_ADDR PET_MASTER_PORT

# 2) 激活 venv(ssh 非交互 shell 不会自动激活)
if [ ! -f "${VENV}/bin/activate" ]; then echo "venv 不存在: ${VENV}"; exit 1; fi
source "${VENV}/bin/activate"

# 3) cd 子模块根 + 校验 config + 起训
cd "${REPO_ROOT}" || { echo "cd 失败: ${REPO_ROOT}"; exit 1; }
if [ ! -f "${CONFIG}" ]; then echo "找不到 config: ${REPO_ROOT}/${CONFIG}"; exit 1; fi
echo "[train_08b_uniform] repo   = ${REPO_ROOT}"
echo "[train_08b_uniform] config = ${CONFIG}"
echo "[train_08b_uniform] python = $(which python)"
echo "[train_08b_uniform] 起训中... 核对下面 single_nodes 打印的 CONFIG_YAML/output_dir 含 08b_subtask_uniform 再走开。"
CONFIG_YAML="${CONFIG}" bash "${ENTRY}"
