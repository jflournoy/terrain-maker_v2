#!/usr/bin/env bash
# Verify a fresh checkout on your own machine: everything the cloud sessions can run, plus
# what they can't (real data, network, Blender, renders). Run from anywhere after `git pull`.
#
#   scripts/verify-local.sh            # sync deps, full tests, mock pins, real-data pins
#   scripts/verify-local.sh --quick    # sync deps + full tests only
#   scripts/verify-local.sh --render   # ...and render the saved preset for a visual check
#   scripts/verify-local.sh --no-real  # skip the real-data pins
#
# Results go to verify-output/<date>-<commit>/ (report.md plus one log per step).
# Real-data pins are compared against a baseline kept in tools/pins/local/ (not committed);
# the first run records it. Accept intended changes with:
#   uv run python tools/pin_combined_render.py record --real
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
QUICK=0 RENDER=0 REAL=1
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --render) RENDER=1 ;;
    --no-real) REAL=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

COMMIT=$(git rev-parse --short HEAD)
OUT="verify-output/$(date +%Y-%m-%d_%H%M)-$COMMIT"
mkdir -p "$OUT"
REPORT="$OUT/report.md"
FAILED=0
FAILED_LOGS=()

{
  echo "# Verification of $COMMIT"
  echo
  echo "- Date: $(date '+%Y-%m-%d %H:%M')"
  echo "- Commit: $(git log -1 --format='%h %s')"
  echo "- Host: $(hostname)"
  echo
  echo "| Step | Result | Time | Log |"
  echo "| --- | --- | --- | --- |"
} > "$REPORT"

# step NAME LOGFILE COMMAND...: run, record pass/fail and duration, keep going on failure
step() {
  local name=$1 log=$2; shift 2
  echo "==> $name"
  local start=$SECONDS
  if "$@" > "$OUT/$log" 2>&1; then
    local result="pass"
  else
    local result="**FAIL**"; FAILED=1; FAILED_LOGS+=("$log")
  fi
  local secs=$((SECONDS - start))
  echo "    $result ($((secs / 60))m$((secs % 60))s) -> $OUT/$log"
  echo "| $name | $result | $((secs / 60))m$((secs % 60))s | [$log]($log) |" >> "$REPORT"
}

step "Sync dependencies" sync.log uv sync
step "Test suite (incl. real-data, network, Blender tests)" pytest.log \
  uv run pytest -rs --junitxml="$OUT/pytest.xml"

if [ "$QUICK" -eq 0 ]; then
  step "Combined render pins (mock data)" pins-mock.log \
    uv run python tools/pin_combined_render.py check
  if [ "$REAL" -eq 1 ]; then
    step "Combined render pins (real data vs local baseline)" pins-real.log \
      uv run python tools/pin_combined_render.py check --real
  fi
fi

if [ "$RENDER" -eq 1 ]; then
  step "Render saved preset" render.log \
    uv run python examples/detroit_combined_render.py \
      @examples/presets/skiing_overhead_dark.args --output-dir "$OUT/render"
fi

{
  for log in ${FAILED_LOGS[@]+"${FAILED_LOGS[@]}"}; do
    echo
    echo "## Failure: $log (last 20 lines)"
    echo
    echo '```text'
    grep -v "^Installed\|^Uninstalled" "$OUT/$log" | tail -20
    echo '```'
  done
  echo
  echo "## Skipped tests"
  echo
  echo '```text'
  grep -E "^SKIPPED" "$OUT/pytest.log" | sed -E 's/^SKIPPED \[[0-9]+\] [^ ]+: //' | sort | uniq -c | sort -rn
  echo '```'
  for log in pins-mock.log pins-real.log; do
    if [ -f "$OUT/$log" ] && grep -q "CHANGED" "$OUT/$log"; then
      echo
      echo "## Changes in $log"
      echo
      echo '```text'
      grep -v "^Installed\|^Uninstalled" "$OUT/$log"
      echo '```'
    fi
  done
  [ -d "$OUT/render" ] && { echo; echo "Render outputs: [render/](render/)"; }
} >> "$REPORT"

echo
echo "Report: $REPORT"
[ "$FAILED" -eq 0 ] && echo "All steps passed." || echo "Some steps FAILED; see the report."
exit "$FAILED"
