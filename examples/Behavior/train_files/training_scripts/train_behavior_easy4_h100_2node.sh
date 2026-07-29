#!/usr/bin/env bash
# Two-node H100 launcher for Easy4 QwenPI_v3 or QwenGR00T (78k, about 5 epochs).
# node99=rank 0, node96=rank 1. Set CONFIG_YAML_OVERRIDE for GR00T.
# Resume: prepare the merged node-local full-state on node99, then launch
# both ranks with IS_RESUME=1.
set -Eeuo pipefail
NODE_RANK="${1:?usage: bash train_behavior_easy4_h100_2node.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "ERROR: NODE_RANK must be 0 (node99) or 1 (node96)"; exit 2; }
cd /nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_models/starvla_dev_20260526_behavior
source /opt/venv/starvla-qwen35/bin/activate

CONFIG_YAML="${CONFIG_YAML_OVERRIDE:-examples/Behavior/train_files/training_config/train_behavior_easy4_qwen35_2b.yaml}"
[ -f "$CONFIG_YAML" ] || { echo "ERROR: missing config: $CONFIG_YAML"; exit 1; }
CONFIG_RUN_ID="$(awk '$1 == "run_id:" { print $2; exit }' "$CONFIG_YAML")"
FRAMEWORK_NAME="$(awk '$1 == "name:" { print $2; exit }' "$CONFIG_YAML")"
[[ "$FRAMEWORK_NAME" == "QwenPI_v3" || "$FRAMEWORK_NAME" == "QwenGR00T" ]] || { echo "ERROR: Easy4 H100 supports QwenPI_v3 or QwenGR00T, got: $FRAMEWORK_NAME"; exit 2; }
[ -n "$CONFIG_RUN_ID" ] || { echo "ERROR: config has no top-level run_id: $CONFIG_YAML"; exit 1; }

BASE_VLM=playground/Pretrained_models/Qwen3.5-2B
[ -d "$BASE_VLM" ] || { echo "ERROR: missing Qwen3.5-2B weights: $BASE_VLM"; exit 1; }
DATA_ROOT=/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data
[ -d "$DATA_ROOT/behavior-1k-easy4" ] || { echo "ERROR: missing Easy4 dataset: $DATA_ROOT/behavior-1k-easy4"; exit 1; }

export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL_OVERRIDE:-PIX}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=10000
export MASTER_ADDR=10.128.99.1
export MASTER_PORT="${MASTER_PORT_OVERRIDE:-29644}"

export FLA_PIN_NORM_AUTOTUNE=1
export FLA_AUTOTUNE_REUSE=1
export HF_SKIP_KWARGS_VALIDATION=1
export TRITON_PRINT_AUTOTUNING=1
export PYTHONUNBUFFERED=1

RUN_ID="${RUN_ID_OVERRIDE:-${CONFIG_RUN_ID}_2node_5ep_h100}"
OUT=results/Checkpoints/$RUN_ID
mkdir -p "$OUT"
LOG=$OUT/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_easy4_h100.log
echo "[easy4-h100] NODE_RANK=$NODE_RANK CONFIG_YAML=$CONFIG_YAML NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL log=$LOG"

EXTRA_ARGS=()
[[ -n "${MAX_TRAIN_STEPS_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--trainer.max_train_steps "$MAX_TRAIN_STEPS_OVERRIDE")
[[ -n "${SAVE_INTERVAL_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--trainer.save_interval "$SAVE_INTERVAL_OVERRIDE")
EXTRA_ARGS+=(--trainer.full_state_save_interval "${FULL_STATE_SAVE_INTERVAL_OVERRIDE:-5000}")
[[ -n "${EVAL_INTERVAL_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--trainer.eval_interval "$EVAL_INTERVAL_OVERRIDE")
[[ -n "${PER_DEVICE_BATCH_SIZE_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--datasets.vla_data.per_device_batch_size "$PER_DEVICE_BATCH_SIZE_OVERRIDE")
[[ "${IS_RESUME:-0}" == "1" ]] && EXTRA_ARGS+=(--trainer.is_resume true)

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_node_local_checkpoint.yaml \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank "$NODE_RANK" \
  --main_process_ip 10.128.99.1 \
  --main_process_port "$MASTER_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_YAML" \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --logging_backend tensorboard \
  --run_root_dir ./results/Checkpoints \
  --run_id "$RUN_ID" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
