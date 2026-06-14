#!/usr/bin/env bash
# 单机多卡训练启动器(在目标 A800 上跑)。做三件事:
#   1) 清掉 pod 注入的多机分布式变量 —— 否则非 master 节点会被顶成 node-rank-1,
#      拿到 rank 8-15,在 dist.barrier() 等不到 master → NCCL system error 挂。
#   2) 激活 venv /opt/venv/starvla-qwen35(ssh 非交互 shell 不会自动激活)。
#   3) cd 到本子模块根,用指定 config 跑 single_nodes/ 入口脚本。
# 路径全部按本脚本位置解析,不写死绝对路径。
#
# 用法:
#   bash examples/Behavior-Skill-GT/train_files/training_scripts/train_singlenode.sh <config文件名或路径>
# 例(2B + 按数据量):
#   bash examples/Behavior-Skill-GT/train_files/training_scripts/train_singlenode.sh train_gt_behavior_skill_easy_qwen35_2b_size_proportional.yaml
# 建议在 tmux 里跑(断 ssh 不掉): tmux new -s wgt 然后执行;Ctrl-b d 脱离。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # .../train_files/training_scripts
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"                  # 子模块根
CFGDIR="examples/Behavior-Skill-GT/train_files/training_config"  # 相对 REPO_ROOT
ENTRY="examples/Behavior-Skill-GT/train_files/training_scripts/single_nodes/run_gt_behavior_skill_easy_singlenode.sh"
VENV=/opt/venv/starvla-qwen35

cfg="$1"
if [ -z "$cfg" ]; then
  echo "用法: bash $(basename "${BASH_SOURCE[0]}") <config文件名或路径>"
  echo "可用 config:"; ls -1 "${REPO_ROOT}/${CFGDIR}" 2>/dev/null | grep -iE "easy.*\.yaml" | sed 's/^/  /'
  exit 1
fi
# 只给文件名 -> 自动补 training_config 前缀;给了路径 -> 原样用
case "$cfg" in
  /*|*/*) config_rel="$cfg" ;;
  *)      config_rel="${CFGDIR}/$cfg" ;;
esac
if [ ! -f "${REPO_ROOT}/${config_rel}" ]; then
  echo "找不到 config: ${REPO_ROOT}/${config_rel}"; exit 1
fi

# 1) 清掉 pod 多机 env,强制单机(singlenode 脚本会自己把 MASTER_ADDR 重设成 127.0.0.1)
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK NNODES NPROC_PER_NODE \
      MASTER_ADDR MASTER_PORT PET_NNODES PET_NODE_RANK PET_NPROC_PER_NODE PET_MASTER_ADDR PET_MASTER_PORT

# 2) 激活 venv
if [ ! -f "${VENV}/bin/activate" ]; then echo "venv 不存在: ${VENV}"; exit 1; fi
source "${VENV}/bin/activate"

# 3) cd 子模块根 + 跑
cd "${REPO_ROOT}" || { echo "cd 失败: ${REPO_ROOT}"; exit 1; }
echo "[train_singlenode] repo   = ${REPO_ROOT}"
echo "[train_singlenode] config = ${config_rel}"
echo "[train_singlenode] python = $(which python)"
echo "[train_singlenode] 起训中... rank 应为 0-7;若出现 rank 8-15 说明多机 env 没清干净。"
CONFIG_YAML="${config_rel}" bash "${ENTRY}"
