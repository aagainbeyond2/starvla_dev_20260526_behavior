#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STARVLA_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DEFAULT_TASKS_JSONL_PATH="${REPO_ROOT}/examples/Behavior/tasks.jsonl"
SKILL_PROMPT_TASKS_JSONL_PATH="${REPO_ROOT}/examples/Behavior-Skill/tasks_skill_prompt.jsonl"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

star_vla_python="${STAR_VLA_PYTHON:-${star_vla_python:-python}}"
TASKS_JSONL_PATH="${TASKS_JSONL_PATH:-${DEFAULT_TASKS_JSONL_PATH}}"
CONFIG_YAML="${CONFIG_YAML:-}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_ID="${RUN_ID:-}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-}"
MODEL_REL_PATH="${MODEL_REL_PATH:-final_model/pytorch_model.pt}"
base_port="${BASE_PORT:-8000}"
SERVER_PROTOCOL="${SERVER_PROTOCOL:-omnigibson_eval}"
EXECUTE_IN_N_STEPS="${EXECUTE_IN_N_STEPS:-30}"
MODEL_ROLLOUT_START_INDEX="${MODEL_ROLLOUT_START_INDEX:-0}"
DEBUG_MODEL_IO="${DEBUG_MODEL_IO:-False}"
USE_STATE="${USE_STATE:-False}"
TEST_NUM="${TEST_NUM:-1}"

yaml_value() {
  local yaml_path="$1"
  local key="$2"
  awk -F': ' -v target="${key}" '$1 == target {print $2; exit}' "${yaml_path}" | tr -d '\r' | sed 's/^"//; s/"$//'
}

resolve_path_from_repo_root() {
  local candidate="$1"
  if [[ "${candidate}" == /* ]]; then
    printf '%s\n' "${candidate}"
  else
    (
      cd "${REPO_ROOT}"
      "${star_vla_python}" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${candidate}"
    )
  fi
}

if (( $# > 0 )); then
  INSTANCE_NAMES=("$@")
else
  INSTANCE_NAMES=("${TASK_NAME:-turning_on_radio}")
fi

# Auto-switch to the skill-prompt task registry when the custom skill task name is used,
# unless the caller explicitly overrides TASKS_JSONL_PATH.
if [[ -z "${TASKS_JSONL_PATH:-}" || "${TASKS_JSONL_PATH}" == "${DEFAULT_TASKS_JSONL_PATH}" ]]; then
  for task_name in "${INSTANCE_NAMES[@]}"; do
    if [[ "${task_name}" == "turning_on_radio_pick_up_from_radio" ]]; then
      TASKS_JSONL_PATH="${SKILL_PROMPT_TASKS_JSONL_PATH}"
      break
    fi
  done
fi

if [[ "${SERVER_PROTOCOL}" != "omnigibson_eval" ]]; then
  echo "Unsupported SERVER_PROTOCOL='${SERVER_PROTOCOL}'. This launcher only supports omnigibson_eval." >&2
  exit 1
fi

if ! command -v "${star_vla_python}" >/dev/null 2>&1 && [[ ! -x "${star_vla_python}" ]]; then
  echo "STAR_VLA_PYTHON is not executable or not in PATH: ${star_vla_python}" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -z "${CONFIG_YAML}" ]]; then
    echo "Please set CONFIG_YAML or MODEL_PATH." >&2
    echo "Example: CONFIG_YAML=examples/Behavior-Skill/train_files/demo_starvla_behavior_skill_easy_qwen35_08b.yaml bash $0 turning_on_radio" >&2
    exit 1
  fi

  if [[ ! -f "${CONFIG_YAML}" ]]; then
    echo "CONFIG_YAML does not exist: ${CONFIG_YAML}" >&2
    exit 1
  fi

  if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="$(yaml_value "${CONFIG_YAML}" run_id)"
  fi
  if [[ -z "${RUN_ROOT_DIR}" ]]; then
    RUN_ROOT_DIR="$(yaml_value "${CONFIG_YAML}" run_root_dir)"
  fi

  if [[ -z "${RUN_ID}" || -z "${RUN_ROOT_DIR}" ]]; then
    echo "Failed to parse run_id/run_root_dir from CONFIG_YAML: ${CONFIG_YAML}" >&2
    exit 1
  fi

  RUN_ROOT_DIR="$(resolve_path_from_repo_root "${RUN_ROOT_DIR}")"
  MODEL_PATH="${RUN_ROOT_DIR}/${RUN_ID}/${MODEL_REL_PATH}"
elif [[ -n "${RUN_ROOT_DIR}" ]]; then
  RUN_ROOT_DIR="$(resolve_path_from_repo_root "${RUN_ROOT_DIR}")"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TASKS_JSONL_PATH}" ]]; then
  echo "TASKS_JSONL_PATH does not exist: ${TASKS_JSONL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/examples/Behavior-Skill/serve_starvla_for_eval.py" ]]; then
  echo "Missing eval-compatible server entry: ${REPO_ROOT}/examples/Behavior-Skill/serve_starvla_for_eval.py" >&2
  exit 1
fi

if ! "${star_vla_python}" -c "import websockets" >/dev/null 2>&1; then
  echo "Missing Python dependency 'websockets' in STAR_VLA_PYTHON: ${star_vla_python}" >&2
  exit 1
fi

if ! "${star_vla_python}" -c "import PIL" >/dev/null 2>&1; then
  echo "Missing Python dependency 'Pillow' in STAR_VLA_PYTHON: ${star_vla_python}" >&2
  exit 1
fi

run_count=0
declare -a used_ports=()
source "${SCRIPT_DIR}/port_utils.sh"

start_service() {
  local gpu_id="$1"
  local ckpt_path="$2"
  local requested_port="$3"
  local task_name="$4"
  local model_parent_dir
  model_parent_dir="$(basename "$(dirname "${ckpt_path}")")"
  local run_dir
  if [[ "${model_parent_dir}" == "checkpoints" || "${model_parent_dir}" == "final_model" ]]; then
    run_dir="$(dirname "$(dirname "${ckpt_path}")")"
  else
    run_dir="$(dirname "${ckpt_path}")"
  fi

  local server_log_dir="${run_dir}/server_logs"
  mkdir -p "${server_log_dir}"

  local ckpt_stem
  ckpt_stem="$(basename "${ckpt_path%.*}")"

  local max_start_attempts=3
  local start_attempt
  for ((start_attempt=1; start_attempt<=max_start_attempts; start_attempt++)); do
    local port
    port="$(find_available_port "${requested_port}")"
    local svc_log="${server_log_dir}/${ckpt_stem}_behavior_skill_eval_server_${task_name}_${port}.log"

    echo "Starting service on GPU ${gpu_id}, port ${port} (requested: ${requested_port}, attempt: ${start_attempt}/${max_start_attempts})" >&2
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" examples/Behavior-Skill/serve_starvla_for_eval.py \
      --ckpt_path "${ckpt_path}" \
      --task-name "${task_name}" \
      --behavior-tasks-jsonl-path "${TASKS_JSONL_PATH}" \
      --port "${port}" \
      --use-bf16 \
      --execute-in-n-steps "${EXECUTE_IN_N_STEPS}" \
      --model-rollout-start-index "${MODEL_ROLLOUT_START_INDEX}" \
      --debug-model-io "${DEBUG_MODEL_IO}" \
      --use-state "${USE_STATE}" \
      > "${svc_log}" 2>&1 &

    local pid=$!
    if wait_for_server "${port}" "${pid}" "${svc_log}"; then
      echo "${pid}:${port}"
      return 0
    fi

    stop_service "${pid}"
    if (( start_attempt < max_start_attempts )); then
      echo "Retry service startup on a new port for task ${task_name}" >&2
    fi
  done

  echo "Failed to start server for task ${task_name} after ${max_start_attempts} attempts" >&2
  return 1
}

stop_service() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping service (PID: ${pid})..."
    kill "${pid}"
    wait "${pid}" 2>/dev/null || true
    echo "Service stopped"
  fi
}

declare -a ALL_SERVICE_PIDS=()
cleanup() {
  for pid in "${ALL_SERVICE_PIDS[@]:-}"; do
    stop_service "${pid}"
  done
}
trap cleanup EXIT INT TERM

IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=${#CUDA_DEVICES[@]}
if [[ ${NUM_GPUS} -eq 0 ]]; then
  CUDA_DEVICES=(0)
  NUM_GPUS=1
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_DEVICES: ${CUDA_DEVICES[*]}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "SERVER_PROTOCOL: ${SERVER_PROTOCOL}"
echo "EXECUTE_IN_N_STEPS: ${EXECUTE_IN_N_STEPS}"
echo "DEBUG_MODEL_IO: ${DEBUG_MODEL_IO}"
echo "TASKS_JSONL_PATH: ${TASKS_JSONL_PATH}"
echo "MODEL_PATH: ${MODEL_PATH}"
if [[ -n "${CONFIG_YAML}" ]]; then
  echo "CONFIG_YAML: ${CONFIG_YAML}"
fi
if [[ -n "${RUN_ID}" ]]; then
  echo "RUN_ID: ${RUN_ID}"
fi

echo "Configuration:"
echo "  - Tasks to serve: ${#INSTANCE_NAMES[@]}"
echo "  - Available GPUs: ${NUM_GPUS} (${CUDA_DEVICES[*]})"
echo "  - Service replicas per task: ${TEST_NUM}"
echo ""

echo "Starting Behavior-Skill eval-compatible model services only"
for i in "${!INSTANCE_NAMES[@]}"; do
  task_name="${INSTANCE_NAMES[i]}"
  for ((run_idx=1; run_idx<=TEST_NUM; run_idx++)); do
    gpu_id="${CUDA_DEVICES[$((run_count % NUM_GPUS))]}"
    requested_port=$((base_port + run_count))
    service_info="$(start_service "${gpu_id}" "${MODEL_PATH}" "${requested_port}" "${task_name}")"
    service_pid="$(echo "${service_info}" | cut -d':' -f1)"
    actual_port="$(echo "${service_info}" | cut -d':' -f2)"

    if [[ -z "${service_pid}" || -z "${actual_port}" ]]; then
      echo "Invalid service info returned for task ${task_name}, aborting..." >&2
      exit 1
    fi

    ALL_SERVICE_PIDS+=("${service_pid}")
    echo "Eval-compatible server ready: task=${task_name}, run=${run_idx}, gpu=${gpu_id}, port=${actual_port}, pid=${service_pid}"
    echo "External eval example: model.host=<server-ip> model.port=${actual_port} task.name=${task_name}"
    run_count=$((run_count + 1))
  done
done

echo ""
echo "Services are running. Press Ctrl+C to stop them."
while true; do
  sleep 3600
done
