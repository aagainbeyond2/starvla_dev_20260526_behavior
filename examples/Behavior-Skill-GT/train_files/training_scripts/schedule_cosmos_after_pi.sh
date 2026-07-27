#!/usr/bin/env bash
# One-shot +7h check, with one retry at +8h from scheduler creation.
set -uo pipefail

REPO=/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_models/starvla_dev_20260526_behavior
WORKFLOW=examples/Behavior-Skill-GT/train_files/training_scripts/run_cosmos_recovery_smoke_then_production.sh
AUTOMATION_DIR="$REPO/results/automation/cosmos2_easy"
SUCCESS_MARKER="$AUTOMATION_DIR/PRODUCTION_STARTED"
START_EPOCH="${1:-$(date +%s)}"
FIRST_EPOCH=$((START_EPOCH + 7 * 3600))
SECOND_EPOCH=$((START_EPOCH + 8 * 3600))
mkdir -p "$AUTOMATION_DIR"
exec >>"$AUTOMATION_DIR/scheduler.log" 2>&1

sleep_until() {
  local target="$1" now delay
  now="$(date +%s)"
  delay=$((target - now))
  ((delay > 0)) && sleep "$delay"
}

echo "[$(date -Is)] scheduled first_epoch=$FIRST_EPOCH second_epoch=$SECOND_EPOCH"
sleep_until "$FIRST_EPOCH"
if bash "$REPO/$WORKFLOW" hour7; then
  echo "[$(date -Is)] hour7 workflow succeeded."
  exit 0
fi

if [[ -f "$SUCCESS_MARKER" ]]; then
  exit 0
fi

echo "[$(date -Is)] hour7 workflow did not complete; retaining fail-closed state for hour8 retry."
sleep_until "$SECOND_EPOCH"
if bash "$REPO/$WORKFLOW" hour8; then
  echo "[$(date -Is)] hour8 workflow succeeded."
  exit 0
fi

echo "[$(date -Is)] hour8 workflow did not complete; no further retries."
exit 1
