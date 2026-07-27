#!/usr/bin/env bash
# Fail-closed two-node Cosmos smoke, exact-resume drill, then production launch.
set -Eeuo pipefail

ATTEMPT_LABEL="${1:?usage: run_cosmos_recovery_smoke_then_production.sh <attempt-label>}"
REPO=/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_models/starvla_dev_20260526_behavior
PEER=node96
PYTHON=/opt/venv/starvla-qwen35/bin/python
LAUNCHER=examples/Behavior-Skill-GT/train_files/training_scripts/train_2node_easy_cosmos_fastio.sh
MERGER=examples/Behavior-Skill-GT/train_files/training_scripts/prepare_2node_full_state_resume.py
PI_RUN=results/Checkpoints/gt_behavior_skill_easy_qwen35_2b_size_proportional_dropout0_140k_fastio
PRODUCTION_RUN=gt_behavior_skill_easy_cosmos2gr00t_2b_size_proportional_dropout0_fastio
AUTOMATION_DIR=results/automation/cosmos2_easy
SUCCESS_MARKER="$AUTOMATION_DIR/PRODUCTION_STARTED"
SMOKE_BATCH_CANDIDATES="${SMOKE_BATCH_CANDIDATES:-4 2}"
EXISTING_SMOKE_RUN_ID="${EXISTING_SMOKE_RUN_ID:-}"
mkdir -p "$REPO/$AUTOMATION_DIR"
cd "$REPO"
exec > >(tee -a "$AUTOMATION_DIR/workflow_${ATTEMPT_LABEL}.log") 2>&1

echo "[$(date -Is)] workflow attempt=$ATTEMPT_LABEL"

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l
}

peer_gpu_process_count() {
  ssh "$PEER" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l"
}

require_idle_and_completed_pi() {
  local local_gpu peer_gpu
  local_gpu="$(gpu_process_count)"
  peer_gpu="$(peer_gpu_process_count)"
  if [[ "$local_gpu" != "0" || "$peer_gpu" != "0" ]]; then
    echo "H100s are not idle: local_gpu=$local_gpu peer_gpu=$peer_gpu"
    return 75
  fi
  if [[ ! -s "$PI_RUN/final_model/pytorch_model.pt" && ! -s "$PI_RUN/final_model/model.safetensors" ]]; then
    echo "PI final model is missing; refusing to start Cosmos."
    return 75
  fi
  if ! grep -aRq "140000/140000" "$PI_RUN"/*.log; then
    echo "PI completion step 140000/140000 is not present in logs."
    return 75
  fi
}

cleanup_smoke() {
  local run_id="$1"
  pkill -f "run_id $run_id" 2>/dev/null || true
  ssh "$PEER" "pkill -f 'run_id $run_id' 2>/dev/null || true" || true
}

run_pair() {
  local run_id="$1" max_steps="$2" batch_size="$3" resume="$4"
  local resume_env=()
  [[ "$resume" == "1" ]] && resume_env+=(IS_RESUME=1)
  local common_env=(
    RUN_ID_OVERRIDE="$run_id"
    MAX_TRAIN_STEPS_OVERRIDE="$max_steps"
    SAVE_INTERVAL_OVERRIDE=1000000
    FULL_STATE_SAVE_INTERVAL_OVERRIDE=2
    EVAL_INTERVAL_OVERRIDE=1000000
    PER_DEVICE_BATCH_SIZE_OVERRIDE="$batch_size"
    MASTER_PORT_OVERRIDE=29636
  )
  local remote_command
  printf -v remote_command "%q " env "${common_env[@]}" "${resume_env[@]}" bash "$LAUNCHER" 1

  set +e
  timeout --signal=TERM 7200 ssh "$PEER" "cd '$REPO' && $remote_command" \
    >"$AUTOMATION_DIR/${run_id}_rank1_steps${max_steps}.log" 2>&1 &
  local peer_pid=$!
  sleep 2
  timeout --signal=TERM 7200 env "${common_env[@]}" "${resume_env[@]}" bash "$LAUNCHER" 0 \
    >"$AUTOMATION_DIR/${run_id}_rank0_steps${max_steps}.log" 2>&1 &
  local local_pid=$!
  wait "$local_pid"
  local local_rc=$?
  wait "$peer_pid"
  local peer_rc=$?
  set -e

  if [[ "$local_rc" != "0" || "$peer_rc" != "0" ]]; then
    echo "Two-node run failed: run_id=$run_id local_rc=$local_rc peer_rc=$peer_rc"
    cleanup_smoke "$run_id"
    return 1
  fi
}

run_resume_drill() {
  local batch_size="$1"
  local run_id="${PRODUCTION_RUN}_resume_smoke_${ATTEMPT_LABEL}_bs${batch_size}"
  if [[ -n "$EXISTING_SMOKE_RUN_ID" ]]; then
    run_id="$EXISTING_SMOKE_RUN_ID"
    echo "Reusing locally complete smoke checkpoint: run_id=$run_id batch_size=$batch_size"
  else
    echo "Starting fresh smoke: run_id=$run_id batch_size=$batch_size"
    if ! run_pair "$run_id" 4 "$batch_size" 0; then
      echo "Fresh smoke failed: run_id=$run_id"
      return 1
    fi
  fi

  if ! "$PYTHON" "$MERGER" \
    --run-dir "results/Checkpoints/$run_id" \
    --peer "$PEER" \
    --step 4; then
    echo "Checkpoint merge failed: run_id=$run_id step=4"
    return 1
  fi

  local completed_resume_log=""
  local candidate_log
  while IFS= read -r candidate_log; do
    if grep -aFq "Restored complete" "$candidate_log" \
      && grep -aFq "Restored data cursor" "$candidate_log" \
      && grep -aFq "Training complete" "$candidate_log"; then
      completed_resume_log="$candidate_log"
    fi
  done < <(find "results/Checkpoints/$run_id" -maxdepth 1 -type f \
    -name "train_node0_*_easy_cosmos_fastio.log" -print | sort)

  if [[ -n "$completed_resume_log" ]] \
    && test -f "results/Checkpoints/$run_id/full_state_checkpoints/steps_000000006/_LOCAL_COMPLETE" \
    && ssh "$PEER" "test -f '$REPO/results/Checkpoints/$run_id/full_state_checkpoints/steps_000000006/_LOCAL_COMPLETE'"; then
    echo "Reusing completed exact-resume drill: log=$completed_resume_log"
    echo "Exact resume drill passed: run_id=$run_id batch_size=$batch_size"
    printf "%s" "$batch_size" >"$AUTOMATION_DIR/validated_batch_size"
    return 0
  fi

  echo "Starting exact resume from step 4 to step 6."
  if ! run_pair "$run_id" 6 "$batch_size" 1; then
    echo "Resume run failed: run_id=$run_id"
    return 1
  fi
  # Rich may wrap "Restored complete training state" across terminal lines.
  # Require both stable prefixes plus the persisted step-6 state instead.
  if ! grep -aFq "Restored complete" "$AUTOMATION_DIR/${run_id}_rank0_steps6.log" \
    || ! grep -aFq "Restored data cursor" "$AUTOMATION_DIR/${run_id}_rank0_steps6.log"; then
    echo "Resume state/cursor success messages are missing: run_id=$run_id"
    return 1
  fi
  if ! test -f "results/Checkpoints/$run_id/full_state_checkpoints/steps_000000006/_LOCAL_COMPLETE"; then
    echo "Local step-6 completion marker is missing: run_id=$run_id"
    return 1
  fi
  if ! ssh "$PEER" "test -f '$REPO/results/Checkpoints/$run_id/full_state_checkpoints/steps_000000006/_LOCAL_COMPLETE'"; then
    echo "Peer step-6 completion marker is missing: run_id=$run_id"
    return 1
  fi
  echo "Exact resume drill passed: run_id=$run_id batch_size=$batch_size"
  printf "%s" "$batch_size" >"$AUTOMATION_DIR/validated_batch_size"
}

production_made_progress() {
  local start_epoch="$1" latest_log
  latest_log="$(find "results/Checkpoints/$PRODUCTION_RUN" -maxdepth 1 -type f \
    -name 'train_node0_*_easy_cosmos_fastio.log' -newermt "@$start_epoch" -print | sort | tail -n 1)"
  [[ -n "$latest_log" ]] || return 1
  grep -aEq '[[:space:]][1-9][0-9]*/547000' "$latest_log"
}

start_production() {
  local batch_size="$1"
  local session0=cosmos2_easy_rank0
  local session1=cosmos2_easy_rank1
  local start_epoch
  start_epoch="$(date +%s)"

  tmux has-session -t "$session0" 2>/dev/null && {
    echo "Production tmux session already exists locally: $session0"
    return 1
  }
  ssh "$PEER" "tmux has-session -t '$session1' 2>/dev/null" && {
    echo "Production tmux session already exists remotely: $session1"
    return 1
  }

  ssh "$PEER" "cd '$REPO' && tmux new-session -d -s '$session1' \
    'env PER_DEVICE_BATCH_SIZE_OVERRIDE=$batch_size MASTER_PORT_OVERRIDE=29635 bash $LAUNCHER 1'"
  sleep 2
  tmux new-session -d -s "$session0" \
    "env PER_DEVICE_BATCH_SIZE_OVERRIDE=$batch_size MASTER_PORT_OVERRIDE=29635 bash $LAUNCHER 0"

  for _ in $(seq 1 20); do
    sleep 30
    if tmux has-session -t "$session0" 2>/dev/null \
      && ssh "$PEER" "tmux has-session -t '$session1' 2>/dev/null" \
      && [[ "$(gpu_process_count)" -gt 0 ]] \
      && [[ "$(peer_gpu_process_count)" -gt 0 ]] \
      && production_made_progress "$start_epoch"; then
      {
        echo "started_at=$(date -Is)"
        echo "batch_size=$batch_size"
        echo "attempt=$ATTEMPT_LABEL"
      } >"$SUCCESS_MARKER"
      echo "Production Cosmos started on both nodes."
      return 0
    fi
  done

  echo "Production did not become healthy within 10 minutes; stopping only the tmux sessions created here."
  tmux kill-session -t "$session0" 2>/dev/null || true
  ssh "$PEER" "tmux kill-session -t '$session1' 2>/dev/null || true" || true
  return 1
}

require_idle_and_completed_pi
test -d playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
ssh "$PEER" "test -d '$REPO/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World'"

validated_batch=""
for batch_size in $SMOKE_BATCH_CANDIDATES; do
  if run_resume_drill "$batch_size"; then
    validated_batch="$batch_size"
    break
  fi
  echo "Batch size $batch_size smoke failed."
  if ! require_idle_and_completed_pi; then
    echo "H100s are not cleanly idle after failed smoke; stopping."
    exit 1
  fi
done

if [[ -z "$validated_batch" ]]; then
  echo "No candidate batch size passed the exact-resume drill."
  exit 1
fi

require_idle_and_completed_pi
if ! start_production "$validated_batch"; then
  echo "Production launch validation failed."
  exit 1
fi
