#!/usr/bin/env bash
# start_services.sh — bring up the entire MILA test pipeline in one command.
#
# Starts (in order):
#   1. log_retention.py --quiet   (one-shot sweep to keep disk tidy)
#   2. test_bot_headless.py       (the tester-bot HTTP API that drives user msgs)
#   3. prepare_test_runs.py       (copies Notion test-run templates → executions)
#   4. run_test_executions.py     (picks pending executions, spawns the bridge)
#   5. run_test_evaluations.py    (OpenAI-scores completed conversations)
#   6. dashboard_app.py           (live Streamlit dashboard on 127.0.0.1:8502)
#
# Also runs stop_services.sh first to guarantee a clean slate — no orphan
# bridge / chromium processes can survive a restart.
#
# Flags:
#   START_DASHBOARD=0   skip the dashboard (default: 1, i.e. start it)
#   SKIP_RETENTION=1    skip the log-retention sweep (default: run it)
#   DASHBOARD_PORT=N    override the dashboard port (default: 8502)
#   DASHBOARD_HOST=H    override the dashboard bind address (default: 127.0.0.1,
#                       i.e. localhost only — SSH-tunnel from your laptop)

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PYTHON="$DIR/env/bin/python"
STREAMLIT="$DIR/env/bin/streamlit"

START_DASHBOARD="${START_DASHBOARD:-1}"
SKIP_RETENTION="${SKIP_RETENTION:-0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8502}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"

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

# --- Step 6: dashboard --------------------------------------------------------
if [ "$START_DASHBOARD" = "1" ] && [ -x "$STREAMLIT" ] && [ -f "$DIR/dashboard_app.py" ]; then
    echo "   ▶️  Starting dashboard_app.py (Streamlit on $DASHBOARD_HOST:$DASHBOARD_PORT)..."
    nohup "$STREAMLIT" run dashboard_app.py \
        --server.address="$DASHBOARD_HOST" \
        --server.port="$DASHBOARD_PORT" \
        --server.headless=true \
        --browser.gatherUsageStats=false \
        --theme.base=dark \
        > logs/dashboard.log 2>&1 &
    disown
elif [ "$START_DASHBOARD" = "1" ]; then
    echo "   ⚠️  Dashboard skipped: streamlit not found at $STREAMLIT or dashboard_app.py missing"
fi

# Wait for processes to come up
sleep 3

# --- Report -------------------------------------------------------------------
echo ""
echo "✅ Services started! Running processes:"
ps -eo pid,etime,cmd --no-headers \
    | grep -E "test_bot_headless|prepare_test_runs|run_test_executions|run_test_evaluations|streamlit.*dashboard_app" \
    | grep -v grep \
    | awk '{printf "   %5s  up %-10s %s\n", $1, $2, substr($0, index($0,$3), 80)}'

echo ""
echo "📊 Monitor:"
echo "   ./testbot status               # quick CLI snapshot"
echo "   ./testbot tail execute         # tail a service log"
echo "   ./testbot errors               # today's errors"
if [ "$START_DASHBOARD" = "1" ] && [ -x "$STREAMLIT" ]; then
    echo ""
    echo "🌐 Dashboard:  http://$DASHBOARD_HOST:$DASHBOARD_PORT/"
    if [ "$DASHBOARD_HOST" = "127.0.0.1" ]; then
        echo "   SSH tunnel from your laptop:"
        echo "     ssh -i Matti-aupair.pem -L $DASHBOARD_PORT:localhost:$DASHBOARD_PORT ubuntu@<host>"
        echo "   Then open http://localhost:$DASHBOARD_PORT/"
    fi
fi
