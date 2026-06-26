#!/usr/bin/env bash
# stop_services.sh — bring the entire MILA test pipeline down cleanly.
#
# Kills (in order, graceful then forceful):
#   1. run_test_executions.py     — the thing that SPAWNS the bridge, must die
#                                    first so it doesn't immediately fire a new
#                                    bridge subprocess after we kill the
#                                    current one
#   2. playwright_bridge_bot_headless.py  — the currently-running bridge
#   3. chrome-headless-shell      — any chromium children the bridge left
#   4. prepare_test_runs.py
#   5. run_test_evaluations.py
#   6. test_bot_headless.py
#
# Always exits 0 — safe to call as a pre-step inside start_services.sh.

set +e  # don't abort on non-zero kills; we want to try every target

# Patterns killed in order. run_test_executions.py must go FIRST so it can't
# respawn the bridge underneath us.
PATTERNS=(
    "run_test_executions.py"
    "playwright_bridge_bot_headless.py"
    "chrome-headless-shell"
    "prepare_test_runs.py"
    "run_test_evaluations.py"
    "test_bot_headless.py"
)

echo "🛑 Stopping all services..."

# Round 1: polite SIGTERM
for pat in "${PATTERNS[@]}"; do
    pkill -f "$pat" 2>/dev/null
done

sleep 2

# Round 2: if anything's still alive, escalate to SIGKILL
STRAGGLERS=0
for pat in "${PATTERNS[@]}"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
        STRAGGLERS=$((STRAGGLERS + 1))
        pkill -9 -f "$pat" 2>/dev/null
    fi
done

if [ "$STRAGGLERS" -gt 0 ]; then
    sleep 1
fi

# Final audit
REMAINING=$(ps -eo pid,cmd --no-headers 2>/dev/null \
    | grep -E "test_bot_headless|prepare_test_runs|run_test_executions|run_test_evaluations|playwright_bridge_bot_headless|chrome-headless-shell" \
    | grep -v grep || true)

if [ -z "$REMAINING" ]; then
    echo "✅ All services stopped"
    exit 0
fi

echo "⚠️  Some processes still running after SIGKILL — manual cleanup may be needed:"
echo "$REMAINING"
exit 0
