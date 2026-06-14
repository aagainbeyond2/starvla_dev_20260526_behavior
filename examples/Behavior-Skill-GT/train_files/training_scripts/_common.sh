#!/usr/bin/env bash
# ============================================================================
# 共用主体: 被 single_nodes/ 与 multi_nodes/ 下的入口脚本 source(不单独执行)。
# 职责:
#   1) 无论从哪里调用, 定位 repo 根并 cd 过去(yaml/playground/norm_stats 都是
#      repo 根相对路径, train_starvla.py 必须以 repo 根为 CWD 运行)。
#   2) 解析 CONFIG_YAML / data_root / run_id 等, 组装 debug extra_args。
#   3) 把 `accelerate launch` 之后的全部训练参数装进数组 LAUNCH_TAIL[]。
# 不负责的: accelerate 的「机器数/进程数/网络」那几行 —— 单机 vs 多机由各入口脚本自己加。
#
# 入口脚本约定: 设好本节点的网络/NCCL 环境后 source 本文件,
# 然后调用 `accelerate launch --config_file "${accel_ds_config}" <机器flags> "${LAUNCH_TAIL[@]}"`。
# ============================================================================

# ---- 定位 repo 根(本文件固定在 examples/Behavior-Skill-GT/train_files/training_scripts/ 下, 上溯 4 层)----
_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_common_dir}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# ---- 默认配置(均可被环境变量覆盖)----
# 默认主实验 = subtask_uniform; 切 size_proportional 用 CONFIG_YAML=...size_proportional.yaml。
: "${CONFIG_YAML:=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_easy_qwen35_08b_subtask_uniform.yaml}"
config_yaml="${CONFIG_YAML}"
playground_root="${PLAYGROUND_ROOT:-./playground}"
logging_backend="${LOGGING_BACKEND:-tensorboard}"
framework_name="${FRAMEWORK_NAME:-QwenPI_v3}"
# base_vlm: 默认【不】覆盖 yaml(让 yaml 的 framework.qwenvl.base_vlm 生效, 便于模型尺寸消融
#   用专门的 yaml, 不会被脚本静默盖回 0.8B); 只有显式 BASE_VLM=<path> 时才覆盖 yaml。
base_vlm_args=()
if [[ -n "${BASE_VLM:-}" ]]; then
  base_vlm_args+=(--framework.qwenvl.base_vlm "${BASE_VLM}")
fi
# V1.0/easy 真实数据根(此默认值会覆盖 yaml 里的 data_root_dir; 临时换数据用 BEHAVIOR_SKILL_DATA_ROOT=...)
behavior_skill_data_root="${BEHAVIOR_SKILL_DATA_ROOT:-/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data/Behavior_Skill_V1.0/easy}"
# accelerate + deepspeed 配置(distributed_type: DEEPSPEED, deepspeed_multinode_launcher: standard)
accel_ds_config="${ACCEL_DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

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

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
# provenance: 入口脚本 + 本文件 + 配置 一起留档到 output_dir
cp "${BASH_SOURCE[1]}" "${output_dir}/" 2>/dev/null || true   # 调用方入口脚本
cp "${BASH_SOURCE[0]}" "${output_dir}/" 2>/dev/null || true   # 本 _common.sh
cp "${config_yaml}"    "${output_dir}/" 2>/dev/null || true

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

# accelerate launch 之后的训练参数(单机/多机完全共用)。注: --datasets.vla_data.data_root_dir
# 会覆盖 yaml 里的值。空数组展开在 bash 4.4+ set -u 下安全。
LAUNCH_TAIL=(
  starVLA/training/train_starvla.py
  --config_yaml "${config_yaml}"
  --framework.name "${framework_name}"
  "${base_vlm_args[@]}"
  --datasets.vla_data.data_root_dir "${behavior_skill_data_root}"
  --logging_backend "${logging_backend}"
  --run_root_dir "${run_root_dir}"
  --run_id "${run_id}"
  "${extra_args[@]}"
)
