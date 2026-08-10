#!/usr/bin/env bash
# Two-node A800 launcher for the Qwen-initialized GR00T radio/trash Long-only baseline.
set -Eeuo pipefail

NODE_RANK="${1:?usage: bash train_2node_radio_trash_gr00t_qweninit_long_only_a800.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || {
  echo "ERROR: NODE_RANK must be 0 or 1"
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source /opt/venv/starvla-qwen35/bin/activate

CONFIG_YAML="${CONFIG_YAML_OVERRIDE:-examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_radio_trash_long_only_qwen35_gr00t_qweninit_30k.yaml}"
[[ -f "$CONFIG_YAML" ]] || { echo "ERROR: missing config: $CONFIG_YAML"; exit 1; }

yaml_value() { awk -v key="$1" '$1 == key ":" { print $2; exit }' "$CONFIG_YAML"; }
CONFIG_RUN_ID="$(yaml_value run_id)"
FRAMEWORK_NAME="$(yaml_value name)"
ATTN_IMPLEMENTATION="$(yaml_value attn_implementation)"
ACTION_MODE="$(yaml_value action_mode)"
DATA_MIX="$(yaml_value data_mix)"
CONFIG_NORM_STATS="$(yaml_value action_norm_stats_path)"
CONFIG_STEPS="$(yaml_value max_train_steps)"
CONFIG_PRETRAINED="$(yaml_value pretrained_checkpoint)"

[[ "$FRAMEWORK_NAME" == "QwenGR00T" ]] || { echo "ERROR: expected QwenGR00T"; exit 2; }
[[ "$ATTN_IMPLEMENTATION" == "flash_attention_2" ]] || { echo "ERROR: expected flash_attention_2"; exit 2; }
python -c 'import flash_attn' >/dev/null 2>&1 || { echo "ERROR: flash_attn is not installed"; exit 2; }
[[ "$ACTION_MODE" == "b1k_delta" ]] || { echo "ERROR: expected b1k_delta"; exit 2; }
[[ "$DATA_MIX" == "gt_behavior_radio_trash_long_only" ]] || { echo "ERROR: unexpected data_mix: $DATA_MIX"; exit 2; }
[[ "$CONFIG_STEPS" == "30000" ]] || { echo "ERROR: expected 30000 steps"; exit 2; }
[[ -z "$CONFIG_PRETRAINED" ]] || { echo "ERROR: Qwen-init baseline must not load pretrained_checkpoint"; exit 2; }

BASE_VLM=playground/Pretrained_models/Qwen3.5-2B
[[ -d "$BASE_VLM" ]] || { echo "ERROR: missing Qwen3.5-2B: $BASE_VLM"; exit 1; }
DEFAULT_DATA_ROOT="$(cd "$REPO_ROOT/../../training_data" && pwd)"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-$DEFAULT_DATA_ROOT}"
[[ -d "$DATA_ROOT/behavior-1k-easy4" ]] || { echo "ERROR: missing behavior-1k-easy4"; exit 1; }

EXPECTED_NORM_STATS="$REPO_ROOT/examples/Behavior-Skill-GT/norm_stats_easy.json"
[[ -f "$CONFIG_NORM_STATS" ]] || { echo "ERROR: missing norm stats: $CONFIG_NORM_STATS"; exit 1; }
[[ "$(readlink -f "$CONFIG_NORM_STATS")" == "$(readlink -f "$EXPECTED_NORM_STATS")" ]] || {
  echo "ERROR: expected norm_stats_easy.json"
  exit 2
}

export MASTER_ADDR="${TRAIN_MASTER_ADDR:-192.168.112.24}"
export MASTER_PORT="${TRAIN_MASTER_PORT:-29732}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-front1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-front1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_7}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PIX}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=10000
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLA_PIN_NORM_AUTOTUNE=1
export FLA_AUTOTUNE_REUSE=1
export HF_SKIP_KWARGS_VALIDATION=1

RUN_ID="${RUN_ID_OVERRIDE:-${CONFIG_RUN_ID}_2node_24_35_a800_ib}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_ID="${RUN_ID_OVERRIDE:-${CONFIG_RUN_ID}_2node_24_35_a800_ib_smoke}"
  EXTRA_ARGS+=(
    --trainer.max_train_steps "${STEPS:-2}"
    --trainer.save_interval "${SAVE_INTERVAL:-1000}"
    --trainer.eval_interval "${EVAL_INTERVAL:-1000}"
    --trainer.full_state_save false
    --trainer.logging_frequency 1
    --datasets.vla_data.per_device_batch_size "${BATCH_SIZE:-1}"
    --datasets.vla_data.num_workers "${NUM_WORKERS:-1}"
    --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR:-2}"
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[radio-trash-gr00t-qweninit] preflight passed rank=$NODE_RANK run_id=$RUN_ID"
  echo "[radio-trash-gr00t-qweninit] config=$CONFIG_YAML mix=$DATA_MIX steps=$CONFIG_STEPS"
  echo "[radio-trash-gr00t-qweninit] qwen=$BASE_VLM pretrained=none stats=$EXPECTED_NORM_STATS"
  echo "[radio-trash-gr00t-qweninit] master=$MASTER_ADDR:$MASTER_PORT net=$NCCL_SOCKET_IFNAME ib_hca=$NCCL_IB_HCA"
  exit 0
fi

OUT="results/Checkpoints/$RUN_ID"
mkdir -p "$OUT"
LOG="$OUT/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_a800.log"
echo "[radio-trash-gr00t-qweninit] rank=$NODE_RANK run_id=$RUN_ID log=$LOG"
echo "[radio-trash-gr00t-qweninit] qwen=$BASE_VLM pretrained=none stats=$EXPECTED_NORM_STATS"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
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
exit "${PIPESTATUS[0]}"
