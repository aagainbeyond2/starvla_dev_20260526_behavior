#!/usr/bin/env bash
# ============================================================================
# train_multinode.sh —— K8s 多机多卡 starVLA 训练: 一个文件搞定, 零脚本依赖。
#   每个节点(pod)各跑一次本脚本; 平台注入的 PET_* 拓扑变量完成静态 rendezvous(--machine_rank)。
#
# 用法(每个 pod 同样这一条; 短名防传参错):
#   bash train_multinode.sh 2b_size        # 也可 08b_uniform | 08b_size | 2b_uniform | <config路径>
#
# 平台需注入(torch elastic 风格, PET_* 优先, 回退非 PET_*):
#   PET_NNODES / PET_NPROC_PER_NODE / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT
# 有 IB/RoCE 的集群: export NCCL_IB_DISABLE=0 + NCCL_SOCKET_IFNAME=<网卡> NCCL_IB_HCA=<卡列表>; 纯 TCP 设 1。
# 可选 env: DATA_ROOT=...  RUN_ID=...  RUN_ROOT_DIR=...
# ============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
CFGDIR="examples/Behavior-Skill-GT/train_files/training_config"

# 1) config: 短名 -> 完整路径
case "${1:-__EMPTY__}" in
  __EMPTY__)   echo "用法: bash train_multinode.sh {08b_uniform|08b_size|2b_uniform|2b_size|<config>}"; exit 1 ;;
  08b_uniform) CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_08b_subtask_uniform.yaml" ;;
  08b_size)    CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_08b_size_proportional.yaml" ;;
  2b_uniform)  CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_2b_subtask_uniform.yaml" ;;
  2b_size)     CFG="$CFGDIR/train_gt_behavior_skill_easy_qwen35_2b_size_proportional.yaml" ;;
  /*|*/*)      CFG="$1" ;;
  *)           CFG="$CFGDIR/$1" ;;
esac
cd "${REPO_ROOT}" || { echo "cd 失败: ${REPO_ROOT}"; exit 1; }
[ -f "$CFG" ] || { echo "找不到 config: ${REPO_ROOT}/${CFG}"; exit 1; }

# 2) 激活 venv
source /opt/venv/starvla-qwen35/bin/activate || { echo "venv 激活失败: /opt/venv/starvla-qwen35"; exit 1; }

# 3) 多机拓扑(PET_* 优先, 回退非 PET_*; 注意 RANK 是全局 rank, 用 NODE_RANK 当节点 rank)
NNODES="${PET_NNODES:-${NNODES:-1}}"
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MAIN_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MAIN_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
TOTAL_PROCESSES=$(( NNODES * NPROC_PER_NODE )); [ "$TOTAL_PROCESSES" -gt 0 ] || TOTAL_PROCESSES="${WORLD_SIZE:-8}"

# 4) 多机 NCCL(不强制 loopback; IB 默认开, 可被平台覆盖)
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export MASTER_ADDR="${MAIN_ADDR}" MASTER_PORT="${MAIN_PORT}"

# 5) run_id / run_root_dir / data: env > yaml > 默认(与原 _common.sh 同口径)
yaml_value() { awk -F': ' -v k="$1" '$1==k{print $2; exit}' "$CFG" | tr -d '\r' | sed 's/^"//; s/"$//'; }
RUN_ID="${RUN_ID:-$(yaml_value run_id)}";                   RUN_ID="${RUN_ID:-$(basename "${CFG%.*}")}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$(yaml_value run_root_dir)}"; RUN_ROOT_DIR="${RUN_ROOT_DIR:-./playground/Checkpoints}"
DATA_ROOT="${DATA_ROOT:-/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data/Behavior_Skill_V1.0/easy}"

echo "[multinode] NNODES=$NNODES NPROC=$NPROC_PER_NODE NODE_RANK=$NODE_RANK TOTAL=$TOTAL_PROCESSES MASTER=$MAIN_ADDR:$MAIN_PORT"
echo "[multinode] config=$CFG -> output: $RUN_ROOT_DIR/$RUN_ID"
echo "[multinode] python=$(which python)"

# 可选覆盖(默认用 yaml 的 trainer 值): 训练步数 / 存档间隔。例: STEPS=300000 SAVE_INTERVAL=5000 ...
OVR_ARGS=()
[ -n "${STEPS:-}" ]         && OVR_ARGS+=(--trainer.max_train_steps "$STEPS")
[ -n "${SAVE_INTERVAL:-}" ] && OVR_ARGS+=(--trainer.save_interval "$SAVE_INTERVAL")
echo "[multinode] steps=${STEPS:-yaml默认(150000)} save_interval=${SAVE_INTERVAL:-yaml默认(2500)}"

# 日志: 打屏 + 落盘到输出目录, 按 NODE_RANK 分文件(多节点共享 FS, 防互相覆盖)
OUTDIR="$RUN_ROOT_DIR/$RUN_ID"; mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "[multinode] log=$LOGFILE"

# 6) 起训(每节点各一次, 静态 rendezvous); tee 打屏+落盘, PIPESTATUS 保留退出码
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines "$NNODES" \
  --num_processes "$TOTAL_PROCESSES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MAIN_ADDR" \
  --main_process_port "$MAIN_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CFG" \
  --framework.name QwenPI_v3 \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --logging_backend tensorboard \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  "${OVR_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
exit "${PIPESTATUS[0]}"
