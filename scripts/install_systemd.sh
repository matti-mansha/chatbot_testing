#!/usr/bin/env bash
# install_systemd.sh — run the MILA test pipeline under systemd so it survives
# reboots and crashes.
#
#   sudo ./scripts/install_systemd.sh
#
# Creates one unit per service plus a mila-pipeline.target that groups them:
#
#   mila-testbot.service    tester-bot HTTP API (127.0.0.1:8501)
#   mila-prepare.service    watches Notion for triggered test runs
#   mila-execute.service    runs pending executions via the Playwright bridge
#   mila-evaluate.service   scores completed conversations via OpenAI
#
# After installation, use systemctl instead of ./start_services.sh:
#
#   sudo systemctl start|stop|restart mila-pipeline.target
#   systemctl status 'mila-*'
#
# ./testbot status, tail and errors keep working unchanged — stdout is appended
# to the same logs/*.log files start_services.sh used.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Run with sudo: sudo $0" >&2
    exit 1
fi

# Resolve the app directory from this script's location, and the owning user
# from the directory itself — so this works on any host without editing.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
APP_USER="$( stat -c '%U' "$APP_DIR" )"
APP_GROUP="$( stat -c '%G' "$APP_DIR" )"
PY="$APP_DIR/env/bin/python"

echo "  app dir : $APP_DIR"
echo "  user    : $APP_USER:$APP_GROUP"
echo "  python  : $PY"

[ -x "$PY" ] || { echo "❌ venv python not found at $PY — create the venv first" >&2; exit 1; }
[ -f "$APP_DIR/.env" ] || { echo "❌ $APP_DIR/.env not found — create it first" >&2; exit 1; }

install -d -o "$APP_USER" -g "$APP_GROUP" "$APP_DIR/logs"

# --- stop anything started by the old shell script ---------------------------
echo "🧹 Stopping any shell-started services (avoids duplicates)..."
if [ -x "$APP_DIR/stop_services.sh" ]; then
    sudo -u "$APP_USER" "$APP_DIR/stop_services.sh" >/dev/null 2>&1 || true
fi

# --- unit writer -------------------------------------------------------------
# $1 unit name  $2 description  $3 script  $4 log file  $5 extra After=
write_unit() {
    local name="$1" desc="$2" script="$3" logfile="$4" extra_after="${5:-}"
    cat > "/etc/systemd/system/${name}.service" <<UNIT
[Unit]
Description=MILA chatbot testing — ${desc}
After=network-online.target ${extra_after}
Wants=network-online.target
PartOf=mila-pipeline.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PY} ${script}

# Append to the same files start_services.sh wrote, so ./testbot tail keeps working.
StandardOutput=append:${APP_DIR}/logs/${logfile}
StandardError=append:${APP_DIR}/logs/${logfile}

Restart=always
RestartSec=10

# Kill the whole cgroup on stop. This reaps the Playwright bridge and any
# headless Chromium it spawned — the leak ./testbot kill-bridge exists to clean up.
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=mila-pipeline.target
UNIT
    echo "   wrote /etc/systemd/system/${name}.service"
}

# NOTE: deliberately no Requires=/After=multi-user.target here. This target is
# itself WantedBy=multi-user.target, and declaring the reverse relation as well
# forms a dependency cycle that systemd resolves by silently dropping one edge.
cat > /etc/systemd/system/mila-pipeline.target <<'TARGET'
[Unit]
Description=MILA chatbot testing pipeline (all services)

[Install]
WantedBy=multi-user.target
TARGET
echo "   wrote /etc/systemd/system/mila-pipeline.target"

write_unit mila-testbot  "tester-bot API"       test_bot_headless.py    testbot.log
# The executor talks to the tester API, so order it after that unit.
write_unit mila-prepare  "prepare test runs"    prepare_test_runs.py    prepare.log
write_unit mila-execute  "run test executions"  run_test_executions.py  execute.log   mila-testbot.service
write_unit mila-evaluate "run evaluations"      run_test_evaluations.py evaluate.log

# --- nightly log retention as a timer (replaces the cron entry) --------------
cat > /etc/systemd/system/mila-retention.service <<UNIT
[Unit]
Description=MILA chatbot testing — log retention sweep

[Service]
Type=oneshot
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${APP_DIR}
ExecStart=${PY} log_retention.py --quiet
StandardOutput=append:${APP_DIR}/logs/retention.log
StandardError=append:${APP_DIR}/logs/retention.log
UNIT

cat > /etc/systemd/system/mila-retention.timer <<'UNIT'
[Unit]
Description=Nightly MILA log retention sweep

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT
echo "   wrote mila-retention.service + .timer"

# Remove the cron entry the bootstrap installed, so retention does not run twice.
rm -f /etc/cron.d/chatbot-log-retention

systemctl daemon-reload
systemctl enable mila-pipeline.target mila-testbot mila-prepare mila-execute mila-evaluate >/dev/null 2>&1
systemctl enable mila-retention.timer >/dev/null 2>&1
systemctl start mila-pipeline.target mila-testbot mila-prepare mila-execute mila-evaluate
systemctl start mila-retention.timer

sleep 4
echo
echo "✅ Installed and started. Current state:"
systemctl --no-pager --plain list-units 'mila-*' 2>/dev/null | head -12
echo
echo "Manage with:"
echo "   sudo systemctl restart mila-pipeline.target   # restart everything"
echo "   sudo systemctl stop    mila-pipeline.target   # stop everything"
echo "   systemctl status mila-execute                 # one service"
echo "   journalctl -u mila-execute -f                 # follow its output"
echo
echo "⚠️  Do NOT use ./start_services.sh any more — it would start a second copy."
