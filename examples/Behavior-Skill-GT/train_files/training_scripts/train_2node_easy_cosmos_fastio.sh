#!/usr/bin/env bash
# Two-node Cosmos-Predict2 GR00T easy-only training with exact state checkpoints.
# Fresh:  bash train_2node_easy_cosmos_fastio.sh <rank>; node99=0, node96=1.
# Resume: first run prepare_2node_full_state_resume.py on node99, then launch
#         both ranks with IS_RESUME=1.
set -Eeuo pipefail
NODE_RANK="${1:?usage: bash train_2node_easy_cosmos_fastio.sh <0|1>}"
cd /nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_models/starvla_dev_20260526_behavior
source /opt/venv/starvla-qwen35/bin/activate

CONFIG_YAML="${CONFIG_YAML_OVERRIDE:-examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_easy_cosmos2gr00t_2b_size_proportional_dropout0_fastio.yaml}"
[ -f "$CONFIG_YAML" ] || { echo "ERROR: missing config: $CONFIG_YAML"; exit 1; }
CONFIG_RUN_ID="$(awk '$1 == "run_id:" { print $2; exit }' "$CONFIG_YAML")"
[ -n "$CONFIG_RUN_ID" ] || { echo "ERROR: config has no top-level run_id: $CONFIG_YAML"; exit 1; }

WM=playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
[ -d "$WM" ] || { echo "ERROR: missing Cosmos weights: $WM"; exit 1; }

export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL_OVERRIDE:-PIX}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=10000
export MASTER_ADDR=10.128.99.1
export MASTER_PORT="${MASTER_PORT_OVERRIDE:-29635}"

export FLA_PIN_NORM_AUTOTUNE=1
export FLA_AUTOTUNE_REUSE=1
export HF_SKIP_KWARGS_VALIDATION=1
export TRITON_PRINT_AUTOTUNING=1
export PYTHONUNBUFFERED=1

RUN_ID="${RUN_ID_OVERRIDE:-$CONFIG_RUN_ID}"
OUT=results/Checkpoints/$RUN_ID
mkdir -p "$OUT"
LOG=$OUT/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S)_easy_cosmos_fastio.log
echo "[easy-cosmos-fastio] NODE_RANK=$NODE_RANK CONFIG_YAML=$CONFIG_YAML NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL log=$LOG"

EXTRA_ARGS=()
[[ -n "${MAX_TRAIN_STEPS_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--trainer.max_train_steps "$MAX_TRAIN_STEPS_OVERRIDE")
[[ -n "${SAVE_INTERVAL_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--trainer.save_interval "$SAVE_INTERVAL_OVERRIDE")
[[ -n "${FULL_STATE_SAVE_INTERVAL_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--trainer.full_state_save_interval "$FULL_STATE_SAVE_INTERVAL_OVERRIDE")
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
  --datasets.vla_data.data_root_dir /nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data/Behavior_Skill_V1.0/easy \
  --logging_backend tensorboard \
  --run_root_dir ./results/Checkpoints \
  --run_id "$RUN_ID" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
