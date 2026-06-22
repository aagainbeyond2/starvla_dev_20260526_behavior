#!/usr/bin/env bash
# ============================================================================
# train.sh —— 单机多卡 starVLA 训练: 一个文件搞定, 零脚本依赖。
#   不 source _common.sh, 不调 single_nodes/。把原来 train_*.sh→single_nodes/run→_common.sh
#   那条链拍平成这一条 accelerate launch(行为与原 _common.sh 默认路径一致)。
#
# 用法(短名即可, 防传参出错):
#   bash train.sh 2b_size        # 2B  + size_proportional(之前漏训那格)
#   bash train.sh 08b_uniform    # 0.8B + subtask_uniform(基线)
#   bash train.sh 08b_size       # 0.8B + size_proportional
#   bash train.sh 2b_uniform     # 2B  + subtask_uniform
#   bash train.sh <完整config文件名或路径>   # 也支持
# 可选 env 覆盖: NUM_PROCESSES=8  DATA_ROOT=...  RUN_ID=...  RUN_ROOT_DIR=...
# 建议在 tmux 里跑: tmux new -s wgt; Ctrl-b d 脱离。
# ============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
CFGDIR="examples/Behavior-Skill-GT/train_files/training_config"

# 1) config: 短名 -> 完整路径
case "${1:-__EMPTY__}" in
  __EMPTY__)   echo "用法: bash train.sh {08b_uniform|08b_size|2b_uniform|2b_size|<config>}"; exit 1 ;;
  08b_uniform) CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_08b_subtask_uniform.yaml" ;;
  08b_size)    CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_08b_size_proportional.yaml" ;;
  2b_uniform)  CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_2b_subtask_uniform.yaml" ;;
  2b_size)     CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_2b_size_proportional.yaml" ;;
  /*|*/*)      CFG="$1" ;;        # 完整路径
  *)           CFG="$CFGDIR/$1" ;; # 当成 config 目录下的文件名
esac

cd "${REPO_ROOT}" || { echo "cd 失败: ${REPO_ROOT}"; exit 1; }
[ -f "$CFG" ] || { echo "找不到 config: ${REPO_ROOT}/${CFG}"; exit 1; }

# 2) 激活 venv(ssh 非交互 shell 不自动激活)
source /opt/venv/starvla-qwen35/bin/activate || { echo "venv 激活失败: /opt/venv/starvla-qwen35"; exit 1; }

# 3) 清 pod 多机 env(否则非 master 节点 rank 8-15, dist.barrier() 卡死)+ 单机 NCCL(loopback, 关 IB)
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK NNODES NPROC_PER_NODE \
      PET_NNODES PET_NODE_RANK PET_NPROC_PER_NODE PET_MASTER_ADDR PET_MASTER_PORT NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000 NCCL_SOCKET_TIMEOUT_MS=360000

# 4) run_id / run_root_dir: env > yaml > 默认(与原 _common.sh 同口径, 保证输出目录不变)
yaml_value() { awk -F': ' -v k="$1" '$1==k{print $2; exit}' "$CFG" | tr -d '\r' | sed 's/^"//; s/"$//'; }
RUN_ID="${RUN_ID:-$(yaml_value run_id)}";            RUN_ID="${RUN_ID:-$(basename "${CFG%.*}")}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$(yaml_value run_root_dir)}"; RUN_ROOT_DIR="${RUN_ROOT_DIR:-./playground/Checkpoints}"
DATA_ROOT="${DATA_ROOT:-/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data/Behavior_Skill_V1.0/easy}"

echo "[train] repo   = ${REPO_ROOT}"
echo "[train] config = ${CFG}"
echo "[train] output = ${RUN_ROOT_DIR}/${RUN_ID}    <-- 核对这里含目标 config 名再走开"
echo "[train] data   = ${DATA_ROOT}"
echo "[train] python = $(which python)"

# 可选覆盖(默认用 yaml 的 trainer 值): 训练步数 / 存档间隔。
#   例: STEPS=300000 SAVE_INTERVAL=5000 bash train.sh 2b_size
OVR_ARGS=()
[ -n "${STEPS:-}" ]         && OVR_ARGS+=(--trainer.max_train_steps "$STEPS")
[ -n "${SAVE_INTERVAL:-}" ] && OVR_ARGS+=(--trainer.save_interval "$SAVE_INTERVAL")
echo "[train] steps  = ${STEPS:-yaml默认(150000)}   save_interval = ${SAVE_INTERVAL:-yaml默认(2500)}"

# 5) 日志: 同时打屏 + 落盘到输出目录(train_<时间戳>.log, 与 checkpoint/tensorboard 同目录)
OUTDIR="${RUN_ROOT_DIR}/${RUN_ID}"; mkdir -p "$OUTDIR"
LOGFILE="${OUTDIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[train] log    = ${LOGFILE}"

# 6) 起训(单机 8 卡 DeepSpeed ZeRO-2); tee 打屏+落盘, PIPESTATUS 保留训练退出码(项目不用 pipefail)
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES:-8}" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CFG" \
  --framework.name QwenPI_v3 \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --logging_backend tensorboard \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  "${OVR_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
exit "${PIPESTATUS[0]}"
