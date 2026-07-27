#!/usr/bin/env bash
# Two-node (node10 rank 0 + a800 rank 1), 8 GPUs per node.
set -euo pipefail

NODE_RANK="${1:?usage: bash train_behavior_easy4_2node.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "NODE_RANK must be 0 or 1"; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
source /opt/venv/starvla-qwen35/bin/activate

CFG="${CFG:-examples/Behavior/train_files/training_config/train_behavior_easy4_qwen35_2b.yaml}"
MASTER_ADDR="${EASY4_MASTER_ADDR:-192.168.112.10}"
MASTER_PORT="${EASY4_MASTER_PORT:-29660}"
RUN_ID="${RUN_ID:-behavior_easy4_qwen35_2b_abs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/Checkpoints}"
NNODES=2
NPROC_PER_NODE=8
TOTAL_PROCESSES=16

DATA_ROOT="${DATA_ROOT:-/ms/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-front1}"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_6,mlx5_7}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PIX}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

OVR_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  OVR_ARGS+=(
    --trainer.max_train_steps "${STEPS:-5}"
    --trainer.save_interval "${SAVE_INTERVAL:-1000}"
    --trainer.eval_interval "${EVAL_INTERVAL:-1000}"
    --trainer.logging_frequency 1
    --datasets.vla_data.per_device_batch_size "${BATCH_SIZE:-16}"
    --datasets.vla_data.num_workers "${NUM_WORKERS:-2}"
    --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR:-2}"
  )
else
  [[ -n "${STEPS:-}" ]] && OVR_ARGS+=(--trainer.max_train_steps "$STEPS")
  [[ -n "${SAVE_INTERVAL:-}" ]] && OVR_ARGS+=(--trainer.save_interval "$SAVE_INTERVAL")
  [[ -n "${BATCH_SIZE:-}" ]] && OVR_ARGS+=(--datasets.vla_data.per_device_batch_size "$BATCH_SIZE")
  [[ -n "${NUM_WORKERS:-}" ]] && OVR_ARGS+=(--datasets.vla_data.num_workers "$NUM_WORKERS")
fi

OUTDIR="$RUN_ROOT_DIR/$RUN_ID"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "[easy4] rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT data=$DATA_ROOT smoke=${SMOKE:-0}"
echo "[easy4] run=$RUN_ID log=$LOGFILE net=$NCCL_SOCKET_IFNAME ib=$NCCL_IB_HCA"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines "$NNODES" \
  --num_processes "$TOTAL_PROCESSES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CFG" \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --logging_backend tensorboard \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  "${OVR_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
exit "${PIPESTATUS[0]}"
