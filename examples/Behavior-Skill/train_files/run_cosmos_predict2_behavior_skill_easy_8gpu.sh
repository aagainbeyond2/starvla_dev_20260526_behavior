#!/usr/bin/env bash
set -euo pipefail

# Dedicated launcher for the Cosmos-Predict2 (WM4A) Behavior-Skill config.
#
# WHY a separate script: run_qwenpi_v3_behavior_skill_easy_8gpu.sh defaults
# FRAMEWORK_NAME=QwenPI_v3 and unconditionally passes --framework.qwenvl.base_vlm
# pointing at a Qwen VLM, which silently overrides this Cosmos config and runs the
# WRONG framework/backbone. This launcher defaults everything to the Cosmos path.

unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

: "${CONFIG_YAML:=examples/Behavior-Skill/train_files/debug_starvla_behavior_skill_easy_pickup_cosmos_predict2_2b.yaml}"
config_yaml="${CONFIG_YAML}"
playground_root="${PLAYGROUND_ROOT:-../starVLA_playground}"
logging_backend="${LOGGING_BACKEND:-tensorboard}"
# Default to the Cosmos framework, not QwenPI_v3.
framework_name="${FRAMEWORK_NAME:-CosmoPredict2GR00T}"
# Cosmos-Predict2 world-model weights (VAE + T5 + DiT).
base_wm="${BASE_WM:-${playground_root}/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World}"
behavior_skill_data_root="${BEHAVIOR_SKILL_DATA_ROOT:-${playground_root}/behavior-own/easy}"
action_loss_type="${ACTION_LOSS_TYPE:-}"
train_debug_enabled="${TRAIN_DEBUG_ENABLED:-}"
train_debug_dump_every_steps="${TRAIN_DEBUG_DUMP_EVERY_STEPS:-}"
train_debug_max_dump_steps="${TRAIN_DEBUG_MAX_DUMP_STEPS:-}"
train_debug_samples_per_step="${TRAIN_DEBUG_SAMPLES_PER_STEP:-}"
train_debug_save_dir_name="${TRAIN_DEBUG_SAVE_DIR_NAME:-}"
train_debug_save_images="${TRAIN_DEBUG_SAVE_IMAGES:-}"
train_debug_save_arrays="${TRAIN_DEBUG_SAVE_ARRAYS:-}"
train_debug_log_action_rows="${TRAIN_DEBUG_LOG_ACTION_ROWS:-}"
train_debug_num_workers="${TRAIN_DEBUG_NUM_WORKERS:-}"

yaml_value() {
  local key="$1"
  awk -F': ' -v target="${key}" '$1 == target {print $2; exit}' "${config_yaml}" \
    | tr -d '\r' \
    | sed 's/^"//; s/"$//'
}

yaml_run_root_dir="$(yaml_value run_root_dir)"
yaml_run_id="$(yaml_value run_id)"

run_root_dir="${RUN_ROOT_DIR:-${yaml_run_root_dir:-${playground_root}/Checkpoints}}"
run_id="${RUN_ID:-${yaml_run_id:-$(basename "${config_yaml%.*}")}}"
num_processes="${NUM_PROCESSES:-8}"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"

extra_args=()
if [[ -n "${action_loss_type}" ]]; then
  extra_args+=(--framework.action_model.loss_type "${action_loss_type}")
fi
if [[ -n "${train_debug_enabled}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.enabled "${train_debug_enabled}")
fi
if [[ -n "${train_debug_dump_every_steps}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.dump_every_steps "${train_debug_dump_every_steps}")
fi
if [[ -n "${train_debug_max_dump_steps}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.max_dump_steps "${train_debug_max_dump_steps}")
fi
if [[ -n "${train_debug_samples_per_step}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.samples_per_step "${train_debug_samples_per_step}")
fi
if [[ -n "${train_debug_save_dir_name}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.save_dir_name "${train_debug_save_dir_name}")
fi
if [[ -n "${train_debug_save_images}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.save_images "${train_debug_save_images}")
fi
if [[ -n "${train_debug_save_arrays}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.save_arrays "${train_debug_save_arrays}")
fi
if [[ -n "${train_debug_log_action_rows}" ]]; then
  extra_args+=(--datasets.vla_data.debug_dump.log_action_rows "${train_debug_log_action_rows}")
fi
if [[ -n "${train_debug_num_workers}" ]]; then
  extra_args+=(--datasets.vla_data.num_workers "${train_debug_num_workers}")
fi

# Point BOTH world_model.base_wm and the legacy qwenvl.base_vlm fallback at the
# Cosmos weights so the world-model factory resolves correctly either way.
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${framework_name}" \
  --framework.world_model.base_wm "${base_wm}" \
  --framework.qwenvl.base_vlm "${base_wm}" \
  --datasets.vla_data.data_root_dir "${behavior_skill_data_root}" \
  --logging_backend "${logging_backend}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "${extra_args[@]}"
