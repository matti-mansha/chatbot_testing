#!/usr/bin/env bash
# start_services.sh — bring up the entire MILA test pipeline in one command.
#
# Starts (in order):
#   1. log_retention.py --quiet   (one-shot sweep to keep disk tidy)
#   2. test_bot_headless.py       (the tester-bot HTTP API that drives user msgs)
#   3. prepare_test_runs.py       (copies Notion test-run templates → executions)
#   4. run_test_executions.py     (picks pending executions, spawns the bridge)
#   5. run_test_evaluations.py    (OpenAI-scores completed conversations)
#
# Also runs stop_services.sh first to guarantee a clean slate — no orphan
# bridge / chromium processes can survive a restart.
#
# Flags:
#   SKIP_RETENTION=1    skip the log-retention sweep (default: run it)
#
# NOTE: the Streamlit dashboard was retired (redundant — KPIs are delivered as
# JSON to the analytics vendor, not via a UI). Archived in archive/dashboard_app.py.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PYTHON="$DIR/env/bin/python"

SKIP_RETENTION="${SKIP_RETENTION:-0}"

if [ ! -f "$PYTHON" ]; then
    echo "❌ Virtual environment Python not found at: $PYTHON"
    echo "   Please create virtual environment first: python3 -m venv env"
    exit 1
fi

mkdir -p logs

# --- Step 0: guaranteed clean slate (reuses stop_services.sh) -----------------
if [ -x "$DIR/stop_services.sh" ]; then
    echo "🧹 Ensuring clean slate (stop_services.sh)..."
    "$DIR/stop_services.sh" >/dev/null 2>&1 || true
    sleep 1
fi

# --- Step 1: one-shot log retention sweep -------------------------------------
if [ "$SKIP_RETENTION" != "1" ] && [ -f "$DIR/log_retention.py" ]; then
    echo "🧽 Running log retention sweep..."
    "$PYTHON" log_retention.py --quiet >> logs/retention.log 2>&1 || {
        echo "   ⚠️  retention sweep failed (non-fatal, continuing)"
    }
fi

# --- Helper: start a background service with nohup + disown -------------------
start_service() {
    local name="$1"       # human-friendly name
    local script="$2"     # python script (relative path)
    local logfile="$3"    # short stdout/stderr redirect file

    echo "   ▶️  Starting $name..."
    nohup "$PYTHON" "$script" > "logs/$logfile" 2>&1 &
    disown
}

# --- Step 2-5: core services --------------------------------------------------
echo "🚀 Starting services..."

start_service "test_bot_headless.py"    test_bot_headless.py    testbot.log
start_service "prepare_test_runs.py"    prepare_test_runs.py    prepare.log

# Small stagger so test_bot_headless is serving before execute tries to use it
sleep 3

start_service "run_test_executions.py"  run_test_executions.py  execute.log
start_service "run_test_evaluations.py" run_test_evaluations.py evaluate.log

# Wait for processes to come up
sleep 3

# --- Report -------------------------------------------------------------------
echo ""
echo "✅ Services started! Running processes:"
ps -eo pid,etime,cmd --no-headers \
    | grep -E "test_bot_headless|prepare_test_runs|run_test_executions|run_test_evaluations" \
    | grep -v grep \
    | awk '{printf "   %5s  up %-10s %s\n", $1, $2, substr($0, index($0,$3), 80)}'

echo ""
echo "📊 Monitor:"
echo "   ./testbot status               # quick CLI snapshot"
echo "   ./testbot tail execute         # tail a service log"
echo "   ./testbot errors               # today's errors"
echo ""
echo "📥 Analytics for the vendor: logs/analytics/events_YYYYMMDD.jsonl"
if [ -L /etc/nginx/sites-enabled/mila-dashboard ] && systemctl is-active --quiet nginx 2>/dev/null; then
    PUBLIC_IP=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<ec2-public-ip>")
    echo "   Download (via nginx): ${PUBLIC_IP:+https://$PUBLIC_IP/analytics/}"
fi
