#!/usr/bin/env bash
# Two-node A800 launcher for the radio/trash Qwen-initialized PI data ablation.
set -Eeuo pipefail

EXPERIMENT="${1:?usage: bash train_2node_radio_trash_qweninit_ablation_a800.sh <long_only|long_plus_all_easy> <0|1>}"
NODE_RANK="${2:?usage: bash train_2node_radio_trash_qweninit_ablation_a800.sh <long_only|long_plus_all_easy> <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || {
  echo "ERROR: NODE_RANK must be 0 or 1"
  exit 2
}

case "$EXPERIMENT" in
  long_only)
    CONFIG_YAML_DEFAULT=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_radio_trash_long_only_qwen35_pi_qweninit_30k.yaml
    EXPECTED_MIX=gt_behavior_radio_trash_long_only
    EXPECTED_STEPS=30000
    MASTER_ADDR_DEFAULT=192.168.112.24
    ;;
  long_plus_all_easy)
    CONFIG_YAML_DEFAULT=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_radio_trash_long_plus_all_easy_qwen35_pi_qweninit_170k.yaml
    EXPECTED_MIX=gt_behavior_radio_trash_long_plus_all_easy
    EXPECTED_STEPS=170000
    MASTER_ADDR_DEFAULT=192.168.112.27
    ;;
  *)
    echo "ERROR: unknown experiment: $EXPERIMENT"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source /opt/venv/starvla-qwen35/bin/activate

CONFIG_YAML="${CONFIG_YAML_OVERRIDE:-$CONFIG_YAML_DEFAULT}"
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

[[ "$FRAMEWORK_NAME" == "QwenPI_v3" ]] || { echo "ERROR: expected QwenPI_v3"; exit 2; }
[[ "$ATTN_IMPLEMENTATION" == "flash_attention_2" ]] || { echo "ERROR: expected flash_attention_2"; exit 2; }
python -c 'import flash_attn' >/dev/null 2>&1 || { echo "ERROR: flash_attn is not installed"; exit 2; }
[[ "$ACTION_MODE" == "b1k_delta" ]] || { echo "ERROR: expected b1k_delta"; exit 2; }
[[ "$DATA_MIX" == "$EXPECTED_MIX" ]] || { echo "ERROR: unexpected data_mix: $DATA_MIX"; exit 2; }
[[ "$CONFIG_STEPS" == "$EXPECTED_STEPS" ]] || { echo "ERROR: unexpected max_train_steps: $CONFIG_STEPS"; exit 2; }
[[ -z "$CONFIG_PRETRAINED" ]] || { echo "ERROR: Qwen-init ablation must not load pretrained_checkpoint"; exit 2; }

BASE_VLM=playground/Pretrained_models/Qwen3.5-2B
[[ -d "$BASE_VLM" ]] || { echo "ERROR: missing Qwen3.5-2B: $BASE_VLM"; exit 1; }
DEFAULT_DATA_ROOT="$(cd "$REPO_ROOT/../../training_data" && pwd)"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-$DEFAULT_DATA_ROOT}"
[[ -d "$DATA_ROOT/behavior-1k-easy4" ]] || { echo "ERROR: missing behavior-1k-easy4"; exit 1; }
if [[ "$EXPERIMENT" == "long_plus_all_easy" ]]; then
  EASY_ROOT="$DATA_ROOT/Behavior_Skill_V1.0/easy"
  [[ -d "$EASY_ROOT" ]] || { echo "ERROR: missing Easy Skill root: $EASY_ROOT"; exit 1; }
  EASY_COUNT="$(find "$EASY_ROOT" -mindepth 3 -maxdepth 3 -type d -name meta | wc -l)"
  [[ "$EASY_COUNT" == "34" ]] || { echo "ERROR: expected 34 Easy Skill datasets, found $EASY_COUNT"; exit 2; }
fi

EXPECTED_NORM_STATS="$REPO_ROOT/examples/Behavior-Skill-GT/norm_stats_easy.json"
[[ -f "$CONFIG_NORM_STATS" ]] || { echo "ERROR: missing norm stats: $CONFIG_NORM_STATS"; exit 1; }
[[ "$(readlink -f "$CONFIG_NORM_STATS")" == "$(readlink -f "$EXPECTED_NORM_STATS")" ]] || {
  echo "ERROR: expected norm_stats_easy.json"
  exit 2
}

export MASTER_ADDR="${TRAIN_MASTER_ADDR:-$MASTER_ADDR_DEFAULT}"
export MASTER_PORT="${TRAIN_MASTER_PORT:-29710}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-front1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-front1}"
# Use containers with RDMA verbs passed through; survey-112-27 has a down
# mlx5_6 rail, so the common safe default intentionally excludes that HCA.
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

RUN_ID="${RUN_ID_OVERRIDE:-${CONFIG_RUN_ID}_2node_a800}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_ID="${RUN_ID_OVERRIDE:-${CONFIG_RUN_ID}_2node_a800_smoke}"
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
  echo "[qweninit-ablation] preflight passed experiment=$EXPERIMENT rank=$NODE_RANK run_id=$RUN_ID"
  echo "[qweninit-ablation] config=$CONFIG_YAML mix=$DATA_MIX steps=$CONFIG_STEPS"
  echo "[qweninit-ablation] qwen=$BASE_VLM pretrained=none stats=$EXPECTED_NORM_STATS"
  echo "[qweninit-ablation] master=$MASTER_ADDR:$MASTER_PORT net=$NCCL_SOCKET_IFNAME ib_disable=$NCCL_IB_DISABLE ib_hca=$NCCL_IB_HCA gdr=$NCCL_NET_GDR_LEVEL"
  exit 0
fi

OUT="results/Checkpoints/$RUN_ID"
mkdir -p "$OUT"
LOG="$OUT/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_a800.log"
echo "[qweninit-ablation] experiment=$EXPERIMENT rank=$NODE_RANK run_id=$RUN_ID log=$LOG"
echo "[qweninit-ablation] qwen=$BASE_VLM pretrained=none stats=$EXPECTED_NORM_STATS"

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
