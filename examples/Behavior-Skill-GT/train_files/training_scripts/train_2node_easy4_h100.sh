#!/usr/bin/env bash
# Behavior-Skill-GT entry for Easy4 QwenPI_v3 or QwenGR00T (78k, about 5 epochs).
# Run directly on the H100 hosts: node99=rank 0, node96=rank 1; no Docker is used.
# Set CONFIG_YAML_OVERRIDE to the sibling QwenGR00T config when training GR00T.
set -Eeuo pipefail

NODE_RANK="${1:?usage: bash train_2node_easy4_h100.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || {
  echo "ERROR: NODE_RANK must be 0 (node99) or 1 (node96)"
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source /opt/venv/starvla-qwen35/bin/activate

DEFAULT_CONFIG=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_easy4_qwen35_2b_size_proportional_dropout0_78k.yaml
CONFIG_YAML="${CONFIG_YAML_OVERRIDE:-$DEFAULT_CONFIG}"
[ -f "$CONFIG_YAML" ] || { echo "ERROR: missing config: $CONFIG_YAML"; exit 1; }

CONFIG_RUN_ID="$(awk '$1 == "run_id:" { print $2; exit }' "$CONFIG_YAML")"
FRAMEWORK_NAME="$(awk '$1 == "name:" { print $2; exit }' "$CONFIG_YAML")"
ACTION_MODE="$(awk '$1 == "action_mode:" { print $2; exit }' "$CONFIG_YAML")"
CONFIG_NORM_STATS="$(awk '$1 == "action_norm_stats_path:" { print $2; exit }' "$CONFIG_YAML")"
[[ "$FRAMEWORK_NAME" == "QwenPI_v3" || "$FRAMEWORK_NAME" == "QwenGR00T" ]] || {
  echo "ERROR: Easy4 H100 supports QwenPI_v3 or QwenGR00T, got: $FRAMEWORK_NAME"
  exit 2
}
[ -n "$CONFIG_RUN_ID" ] || { echo "ERROR: config has no top-level run_id: $CONFIG_YAML"; exit 1; }

BASE_VLM=playground/Pretrained_models/Qwen3.5-2B
[ -d "$BASE_VLM" ] || { echo "ERROR: missing Qwen3.5-2B weights: $BASE_VLM"; exit 1; }

DEFAULT_DATA_ROOT="$(cd "$REPO_ROOT/../../training_data" && pwd)"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-$DEFAULT_DATA_ROOT}"
EASY4_ROOT="$DATA_ROOT/behavior-1k-easy4"
NORM_STATS="$EASY4_ROOT/meta/norm_stats_easy4_b1k_delta.json"
[ -d "$EASY4_ROOT" ] || { echo "ERROR: missing Easy4 dataset: $EASY4_ROOT"; exit 1; }
[ -f "$NORM_STATS" ] || { echo "ERROR: missing formal H100 delta stats: $NORM_STATS"; exit 1; }
[[ "$ACTION_MODE" == "b1k_delta" ]] || { echo "ERROR: action_mode must be b1k_delta, got: $ACTION_MODE"; exit 2; }
[ -f "$CONFIG_NORM_STATS" ] || { echo "ERROR: config stats path is missing: $CONFIG_NORM_STATS"; exit 1; }
CONFIG_NORM_STATS_REAL="$(readlink -f "$CONFIG_NORM_STATS")"
NORM_STATS_REAL="$(readlink -f "$NORM_STATS")"
[[ "$CONFIG_NORM_STATS_REAL" == "$NORM_STATS_REAL" ]] || {
  echo "ERROR: config does not reference the formal H100 stats: $CONFIG_NORM_STATS_REAL"
  exit 2
}

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
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[easy4-h100] preflight passed: rank=$NODE_RANK framework=$FRAMEWORK_NAME config=$CONFIG_YAML stats=$NORM_STATS_REAL run_id=$RUN_ID"
  exit 0
fi
OUT="results/Checkpoints/$RUN_ID"
mkdir -p "$OUT"
LOG="$OUT/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_easy4_h100.log"
echo "[easy4-h100] NODE_RANK=$NODE_RANK FRAMEWORK=$FRAMEWORK_NAME CONFIG_YAML=$CONFIG_YAML"
echo "[easy4-h100] DATA_ROOT=$DATA_ROOT NORM_STATS=$NORM_STATS log=$LOG"

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
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_YAML" \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --logging_backend tensorboard \
  --run_root_dir ./results/Checkpoints \
  --run_id "$RUN_ID" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
