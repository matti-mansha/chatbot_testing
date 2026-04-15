"""
MILA Test Control — a live dashboard for the chatbot testing pipeline.

Reads the structured logs under logs/, the running process table, and the
currently-spawned Playwright bridge subprocess to render a real-time view
of the four-service pipeline (test_bot_headless, prepare_test_runs,
run_test_executions, run_test_evaluations) plus the bridge it spawns.

Features:
    * Service health pills with pulsing "alive" indicator
    * Currently-running bridge card: test case, persona, attempt,
      turn N/MAX, elapsed time, latest tester/Mila replies
    * Queue depth (pending / running / completed / failed today)
    * Today's outcomes donut + KPI sparkline
    * Recent-history table with expandable conversation preview
    * Live log tail with service selector + regex filter
    * Failures drill-down grouped by reason
    * Kill-bridge button (with confirmation)
    * Auto-refresh every 5 seconds (JS-driven, preserves Streamlit state
      between refreshes via st.session_state)

Run locally:
    env/bin/streamlit run dashboard_app.py --server.address=127.0.0.1 --server.port=8502

Then SSH-tunnel on your laptop:
    ssh -i Matti-aupair.pem -L 8502:localhost:8502 ubuntu@<host>
And open http://localhost:8502 in a browser.
"""
from __future__ import annotations

import os
import re
import json
import subprocess
import signal
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import textwrap


def html(raw: str) -> None:
    """
    Render HTML via st.markdown(unsafe_allow_html=True) without the
    CommonMark "indented code block" trap.

    Streamlit's markdown pipeline is CommonMark, which treats any line
    that is whitespace-only as a blank line. Inside an HTML block of
    type 6 (starts with <div>, etc.), a blank line ENDS the HTML block.
    Anything that follows with 4+ leading spaces is then rendered as an
    indented code block — which is why <code>&lt;/div&gt;</code> literal
    strings were leaking into the hero.

    This helper:
        1. textwrap.dedent()s the string (kills common leading whitespace)
        2. Strips every purely whitespace line (so empty f-string
           substitutions can't open a code block)
        3. Left-justifies (strips leading whitespace on each remaining
           line) so no line has 4+ leading spaces — belt-and-braces

    The result is a sequence of content-only lines that CommonMark
    treats as a single continuous HTML block.
    """
    dedented = textwrap.dedent(raw)
    lines = [ln.lstrip() for ln in dedented.splitlines() if ln.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)

# streamlit-autorefresh is the clean way to do partial-rerun-based auto-refresh
# (no page reload, no flicker, preserves scroll). Fall back to a full page
# reload via a components.v1.html iframe if the package isn't installed.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
DIAG_DIR = BASE_DIR / "diagnostics"

AUTO_REFRESH_MS = 5000  # 5 seconds
MAX_TAIL_LINES = 500
TAIL_LINES_DEFAULT = 120

# Services we watch. Keys are human-friendly names; values are (process
# substring to match in `ps -ef`, log-file base name).
SERVICES: Dict[str, Tuple[str, str]] = {
    "tester_bot": ("test_bot_headless.py", "test_bot_headless"),
    "prepare": ("prepare_test_runs.py", "prepare_test_runs"),
    "execute": ("run_test_executions.py", "test_execution"),
    "evaluate": ("run_test_evaluations.py", "test_evaluation"),
}

# Extra log file for the spawned bridge subprocess (not a service).
BRIDGE_LOG_BASENAME = "playwright_bridge"

# =============================================================================
# STREAMLIT PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="MILA Test Control",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Custom CSS: dark theme polish, pulsing status dots, hero header ----
st.markdown(
    """
<style>
/* Tighten Streamlit's default margins */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1d2e 0%, #2d1b3d 50%, #1a3a5c 100%);
    border-radius: 16px;
    padding: 22px 30px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
.hero-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.5px;
    background: linear-gradient(90deg, #fbbf24 0%, #f472b6 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #9ca3af;
    font-size: 13px;
    margin-top: 4px;
    font-variant-numeric: tabular-nums;
}

/* Service status pills */
.status-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 13px;
    font-weight: 500;
    color: #d1d5db;
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.dot-alive {
    background: #22c55e;
    box-shadow: 0 0 0 0 rgba(34,197,94,0.7);
    animation: pulseGreen 2s infinite;
}
.dot-dead {
    background: #ef4444;
    box-shadow: 0 0 8px rgba(239,68,68,0.5);
}
.dot-unknown {
    background: #6b7280;
}
@keyframes pulseGreen {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.7); }
    70%  { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

/* Cards */
.card {
    background: #151822;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.card-active {
    border: 1px solid rgba(34,197,94,0.35);
    box-shadow: 0 0 0 0 rgba(34,197,94,0.35);
    animation: pulseBorder 2.2s infinite;
}
@keyframes pulseBorder {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.35); }
    70%  { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
.card-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #9ca3af;
    font-weight: 600;
    margin-bottom: 10px;
}
.big-name {
    font-size: 22px;
    font-weight: 700;
    color: #f3f4f6;
    margin-bottom: 2px;
}
.muted {
    color: #9ca3af;
    font-size: 13px;
}
.kv {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
    border-bottom: 1px dashed rgba(255,255,255,0.05);
}
.kv:last-child { border-bottom: none; }
.kv-k { color: #9ca3af; }
.kv-v { color: #f3f4f6; font-variant-numeric: tabular-nums; font-weight: 600; }

/* Metric strip */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 14px;
}
.metric-cell {
    background: #151822;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 16px;
}
.metric-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9ca3af;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #f3f4f6;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}
.metric-sub {
    margin-top: 4px;
    font-size: 11px;
    color: #6b7280;
}

/* Status badges for table rows */
.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.badge-ok    { background: rgba(34,197,94,0.18);  color: #4ade80; }
.badge-fail  { background: rgba(239,68,68,0.18);  color: #f87171; }
.badge-slow  { background: rgba(250,204,21,0.18); color: #facc15; }
.badge-run   { background: rgba(96,165,250,0.18); color: #60a5fa; }

/* Log tail */
.log-box {
    background: #0b0d14;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: 'SF Mono', 'Fira Code', Menlo, Consolas, monospace;
    font-size: 11.5px;
    color: #cbd5e1;
    white-space: pre-wrap;
    max-height: 480px;
    overflow-y: auto;
    line-height: 1.45;
}
.log-err  { color: #f87171; }
.log-warn { color: #facc15; }
.log-ok   { color: #4ade80; }
.log-info { color: #cbd5e1; }

/* Progress bar */
.progress-outer {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0 4px 0;
}
.progress-inner {
    height: 100%;
    background: linear-gradient(90deg, #4ade80 0%, #60a5fa 100%);
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Remove Streamlit's default header/footer clutter */
#MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LAYER
# =============================================================================

def today_log(basename: str) -> Path:
    return LOG_DIR / f"{basename}_{datetime.now().strftime('%Y%m%d')}.log"


def tail_file(path: Path, n: int = TAIL_LINES_DEFAULT) -> List[str]:
    """Return the last n lines of a text file without loading it all into memory."""
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
            lines = data.decode("utf-8", errors="replace").splitlines()
            return lines[-n:]
    except Exception:
        return []


def ps_processes() -> List[Dict[str, str]]:
    """Return a list of processes with fields from `ps -ef`."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etime,stat,cmd", "--no-headers"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        rows.append(
            {"pid": parts[0], "etime": parts[1], "stat": parts[2], "cmd": parts[3]}
        )
    return rows


def service_state() -> Dict[str, Dict[str, Any]]:
    """Return {service_name: {alive, pid, etime, cmd}}."""
    procs = ps_processes()
    state: Dict[str, Dict[str, Any]] = {}
    for name, (needle, _log_base) in SERVICES.items():
        match = None
        for p in procs:
            if needle in p["cmd"] and "grep" not in p["cmd"]:
                match = p
                break
        if match:
            state[name] = {
                "alive": True,
                "pid": match["pid"],
                "etime": match["etime"],
                "cmd": match["cmd"],
            }
        else:
            state[name] = {"alive": False, "pid": None, "etime": None, "cmd": None}
    return state


def bridge_process() -> Optional[Dict[str, str]]:
    """Return info about the currently-running playwright bridge subprocess."""
    for p in ps_processes():
        if "playwright_bridge_bot_headless.py" in p["cmd"] and "grep" not in p["cmd"]:
            return p
    return None


LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*"
    r"(?P<svc>[^|]+?)\s*\|\s*(?P<lvl>[^|]+?)\s*\|\s*"
    r"(?P<loc>[^|]+?)\s*\|\s*(?P<msg>.*)$"
)


def parse_log_line(line: str) -> Optional[Dict[str, str]]:
    m = LOG_LINE_RE.match(line)
    if not m:
        return None
    return m.groupdict()


@dataclass
class BridgeStatus:
    running: bool = False
    test_case: str = ""
    persona: str = ""
    run_number: str = ""
    attempt: int = 0
    attempt_total: int = 0
    turn: int = 0
    max_turns: int = 15
    elapsed_sec: float = 0.0
    last_mila_reply: str = ""
    last_tester_reply: str = ""
    started_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    mila_reject_attempt: int = 0
    general_restart_attempt: int = 0


_ETIME_RE = re.compile(
    r"(?:(?P<d>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)"
)


def parse_etime_to_seconds(etime: str) -> int:
    """Convert `ps -o etime` output (e.g. 01:37, 02:15:04, 1-12:34:56) to seconds."""
    if not etime:
        return 0
    m = _ETIME_RE.match(etime.strip())
    if not m:
        return 0
    d = int(m.group("d") or 0)
    h = int(m.group("h") or 0)
    mm = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    return d * 86400 + h * 3600 + mm * 60 + s


def _latest_run_number_from_execute_log(max_lines: int = 400) -> str:
    """Grab the most recent "Test run number: TR1.TCXX.N" from test_execution log."""
    path = today_log("test_execution")
    lines = tail_file(path, n=max_lines)
    for line in reversed(lines):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        m = re.search(r"Test run number:\s+(\S+)", parsed["msg"])
        if m:
            return m.group(1).strip()
    return ""


def _latest_persona_from_execute_log(max_lines: int = 400) -> str:
    path = today_log("test_execution")
    lines = tail_file(path, n=max_lines)
    for line in reversed(lines):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        m = re.search(r"Persona:\s+(.+)$", parsed["msg"])
        if m:
            return m.group(1).strip()
    return ""


def parse_bridge_status(max_lines: int = 600) -> BridgeStatus:
    """Scrape the latest bridge activity from playwright_bridge_<date>.log."""
    path = today_log(BRIDGE_LOG_BASENAME)
    status = BridgeStatus()
    lines = tail_file(path, n=max_lines)

    last_turn_set = False
    last_mila_set = False
    started_at_set = False
    attempt_set = False
    test_case_set = False

    for line in reversed(lines or []):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        msg = parsed["msg"]
        try:
            ts = datetime.strptime(parsed["ts"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = None
        if status.last_event_at is None and ts is not None:
            status.last_event_at = ts

        if not last_turn_set:
            m = re.search(r"TURN (\d+)/(\d+)", msg)
            if m:
                status.turn = int(m.group(1))
                status.max_turns = int(m.group(2))
                last_turn_set = True

        # Mila reply preview — the log format after the final `|` has its
        # leading whitespace stripped by the pipe-separator regex, so the
        # message starts with "Content: …" (no leading spaces).
        if not last_mila_set and msg.startswith("Content:"):
            status.last_mila_reply = msg[len("Content:"):].strip()
            last_mila_set = True

        # Match STARTING or RESTARTING TEST EXECUTION specifically — NOT
        # the bare "Attempt N/M" which also appears in the tester-API send
        # retry counter and would cause us to show "1/3" instead of the
        # real restart-attempt counter.
        if not attempt_set:
            m = re.search(
                r"(?:RE)?STARTING TEST EXECUTION\s*\(Attempt (\d+)/(\d+)"
                r"(?:, general=(\d+)/\d+, mila-reject=(\d+)/\d+)?",
                msg,
            )
            if m:
                status.attempt = int(m.group(1))
                status.attempt_total = int(m.group(2))
                if m.group(3):
                    status.general_restart_attempt = int(m.group(3))
                if m.group(4):
                    status.mila_reject_attempt = int(m.group(4))
                attempt_set = True

        if not test_case_set:
            m = re.search(r"Test case:\s+(.+)$", msg)
            if m:
                status.test_case = m.group(1).strip()
                test_case_set = True

        if not started_at_set and "STARTING TEST EXECUTION" in msg and ts:
            status.started_at = ts
            started_at_set = True

        if last_turn_set and last_mila_set and attempt_set and test_case_set and started_at_set:
            break

    # Detect running vs finished by checking the process table
    bproc = bridge_process()
    status.running = bproc is not None

    # Fall back to the cmdline if the log hasn't caught up with a fresh spawn
    if bproc and not status.test_case:
        cmd = bproc["cmd"]
        m = re.search(r"playwright_bridge_bot_headless\.py\s+(.*)$", cmd)
        if m:
            status.test_case = m.group(1).split(" ", 1)[0][:80]

    # Cross-reference with execute log for the run number + persona (those
    # live in test_execution_<date>.log, not the bridge log)
    if status.running and not status.run_number:
        status.run_number = _latest_run_number_from_execute_log()
    if status.running and not status.persona:
        status.persona = _latest_persona_from_execute_log()

    # Elapsed-time computation: prefer the log-derived started_at, but fall
    # back to the process's `ps etime` if the log parser didn't find a
    # STARTING marker (e.g. a very fresh bridge where the log line hasn't
    # flushed yet).
    if status.started_at:
        ref = datetime.now() if status.running else (status.last_event_at or datetime.now())
        status.elapsed_sec = max(0.0, (ref - status.started_at).total_seconds())
    elif status.running and bproc is not None:
        status.elapsed_sec = float(parse_etime_to_seconds(bproc.get("etime", "")))

    return status


@dataclass
class ExecutionRecord:
    ts: datetime
    test_case: str
    run_number: str
    status: str  # "passed" | "failed" | "unknown"
    duration_sec: Optional[float] = None
    failure_reason: Optional[str] = None


def parse_recent_executions(max_lines: int = 4000) -> List[ExecutionRecord]:
    """Scan test_execution_<date>.log for completed / failed executions."""
    path = today_log("test_execution")
    records: List[ExecutionRecord] = []
    lines = tail_file(path, n=max_lines)
    if not lines:
        return records

    pending_test_case = ""
    pending_run_number = ""
    pending_start: Optional[datetime] = None
    pending_failure: Optional[str] = None

    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        msg = parsed["msg"]
        try:
            ts = datetime.strptime(parsed["ts"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = None

        m = re.search(r"Test case name:\s+(.+)$", msg)
        if m:
            pending_test_case = m.group(1).strip()
            continue
        m = re.search(r"Test run number:\s+(.+)$", msg)
        if m:
            pending_run_number = m.group(1).strip()
            continue
        m = re.search(r"▶ Processing execution", msg)
        if m and ts:
            pending_start = ts
            pending_failure = None
            continue
        m = re.search(r"Bridge reported failure .* '([^']+)'", msg)
        if m:
            pending_failure = m.group(1)
            continue
        if "BRIDGE_FAILURE_REASON=" in msg:
            m = re.search(r"BRIDGE_FAILURE_REASON=(\S+)", msg)
            if m:
                pending_failure = m.group(1)
            continue
        if "Execution" in msg and "completed successfully" in msg and ts:
            dur = (ts - pending_start).total_seconds() if pending_start else None
            records.append(
                ExecutionRecord(
                    ts=ts,
                    test_case=pending_test_case or "—",
                    run_number=pending_run_number or "",
                    status="passed",
                    duration_sec=dur,
                )
            )
            pending_test_case = pending_run_number = ""
            pending_start = None
            continue
        if "Execution" in msg and "failed" in msg and ts:
            dur = (ts - pending_start).total_seconds() if pending_start else None
            records.append(
                ExecutionRecord(
                    ts=ts,
                    test_case=pending_test_case or "—",
                    run_number=pending_run_number or "",
                    status="failed",
                    duration_sec=dur,
                    failure_reason=pending_failure or "unknown",
                )
            )
            pending_test_case = pending_run_number = ""
            pending_start = None
            pending_failure = None
            continue

    return records


def pending_count_from_execute_log() -> Optional[int]:
    """Latest 'Found N pending executions' count from the execute log."""
    path = today_log("test_execution")
    lines = tail_file(path, n=200)
    for line in reversed(lines):
        m = re.search(r"Found (\d+) pending executions", line)
        if m:
            return int(m.group(1))
    return None


def log_dir_size_bytes() -> int:
    total = 0
    if LOG_DIR.exists():
        for p in LOG_DIR.glob("*.log*"):
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def diag_dir_size_bytes() -> int:
    total = 0
    if DIAG_DIR.exists():
        for p in DIAG_DIR.iterdir():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore
    return f"{n:.1f} PB"


def human_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# =============================================================================
# ACTIONS (kill bridge, clear stale, etc.)
# =============================================================================

def kill_bridge() -> Tuple[bool, str]:
    bproc = bridge_process()
    if not bproc:
        return False, "No bridge subprocess running."
    try:
        os.kill(int(bproc["pid"]), signal.SIGKILL)
        return True, f"SIGKILL sent to PID {bproc['pid']}"
    except ProcessLookupError:
        return False, "Process already exited."
    except PermissionError:
        return False, "Permission denied (is the dashboard running as the same user as the services?)"
    except Exception as e:
        return False, f"Kill failed: {e}"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_hero(svc_state: Dict[str, Dict[str, Any]]) -> None:
    pills_html = []
    for name, info in svc_state.items():
        dot = "dot-alive" if info["alive"] else "dot-dead"
        etime = info.get("etime") or "—"
        pills_html.append(
            f'<div class="status-pill"><span class="status-dot {dot}"></span>'
            f'{name}<span class="muted">· {etime}</span></div>'
        )
    bridge = bridge_process()
    bridge_pill = ""
    if bridge:
        bridge_pill = (
            f'<div class="status-pill" style="background: rgba(96,165,250,0.12); border-color: rgba(96,165,250,0.3);">'
            f'<span class="status-dot dot-alive" style="background:#60a5fa;"></span>'
            f'bridge PID {bridge["pid"]}<span class="muted">· {bridge["etime"]}</span></div>'
        )
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.now().strftime("%H:%M:%S")
    refresh_label = (
        f"Auto-refresh · tick {now_local}" if HAS_AUTOREFRESH else f"Full reload · tick {now_local}"
    )
    html(f"""
        <div class="hero">
        <div class="hero-title">🧪 MILA TEST CONTROL</div>
        <div class="hero-sub">{refresh_label} · {now_utc}</div>
        <div class="status-row">{''.join(pills_html)}{bridge_pill}</div>
        </div>
    """)


def render_metric_strip(svc_state: Dict[str, Dict[str, Any]]) -> None:
    records = parse_recent_executions()
    today_passed = sum(1 for r in records if r.status == "passed")
    today_failed = sum(1 for r in records if r.status == "failed")
    today_total = today_passed + today_failed

    pending = pending_count_from_execute_log()
    pending_str = str(pending) if pending is not None else "—"

    avg_dur = "—"
    durs = [r.duration_sec for r in records if r.duration_sec is not None]
    if durs:
        avg_dur = human_duration(sum(durs) / len(durs))

    pass_rate = "—"
    if today_total:
        pass_rate = f"{100 * today_passed / today_total:.0f}%"

    log_size = human_bytes(log_dir_size_bytes())
    diag_size = human_bytes(diag_dir_size_bytes())

    html(f"""
        <div class="metric-strip">
        <div class="metric-cell">
        <div class="metric-label">Passed today</div>
        <div class="metric-value" style="color:#4ade80;">{today_passed}</div>
        <div class="metric-sub">of {today_total} finished</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">Failed today</div>
        <div class="metric-value" style="color:#f87171;">{today_failed}</div>
        <div class="metric-sub">pass rate {pass_rate}</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">Pending</div>
        <div class="metric-value">{pending_str}</div>
        <div class="metric-sub">avg run {avg_dur}</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">Disk</div>
        <div class="metric-value" style="font-size:20px;">{log_size}</div>
        <div class="metric-sub">diagnostics {diag_size}</div>
        </div>
        </div>
    """)


def _escape(text: str) -> str:
    return (text or "").replace("<", "&lt;").replace(">", "&gt;")


def render_bridge_card(status: BridgeStatus) -> None:
    if not status.running:
        html("""
            <div class="card">
            <div class="card-title">CURRENT BRIDGE</div>
            <div class="big-name">💤 Pipeline idle</div>
            <div class="muted">No Playwright bridge subprocess is running right now.
            The next run will fire when prepare/execute sees a pending execution.</div>
            </div>
        """)
        return

    pct = 0
    if status.max_turns:
        pct = min(100, int(100 * status.turn / status.max_turns))
    elapsed = human_duration(status.elapsed_sec)
    test_case = _escape(status.test_case or "(unknown)")
    persona = _escape(status.persona or "")
    last_mila = _escape(status.last_mila_reply)[:400]
    last_tester = _escape(status.last_tester_reply)[:400]

    attempt_line = ""
    if status.attempt and status.attempt_total:
        rej_bit = ""
        if status.mila_reject_attempt:
            rej_bit = f" · mila-reject {status.mila_reject_attempt}"
        if status.general_restart_attempt:
            rej_bit += f" · general-restart {status.general_restart_attempt}"
        attempt_line = (
            f'<div class="kv"><span class="kv-k">Attempt</span>'
            f'<span class="kv-v">{status.attempt}/{status.attempt_total}{rej_bit}</span></div>'
        )

    last_mila_block = last_mila or '<span class="muted">(none yet)</span>'
    html(f"""
        <div class="card card-active">
        <div class="card-title">▶ CURRENT BRIDGE</div>
        <div class="big-name">{test_case}</div>
        <div class="muted">{persona}</div>
        <div class="progress-outer"><div class="progress-inner" style="width: {pct}%;"></div></div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#9ca3af;"><span>Turn {status.turn}/{status.max_turns}</span><span>{pct}%</span></div>
        <div style="margin-top:12px;">
        {attempt_line}
        <div class="kv"><span class="kv-k">Elapsed</span><span class="kv-v">{elapsed}</span></div>
        <div class="kv"><span class="kv-k">Run number</span><span class="kv-v">{_escape(status.run_number) or '—'}</span></div>
        </div>
        <div style="margin-top:12px;">
        <div style="font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">LAST MILA REPLY</div>
        <div style="margin-top:4px; font-size:12.5px; color:#d1d5db; max-height:100px; overflow-y:auto;">{last_mila_block}</div>
        </div>
        </div>
    """)

    # Kill button (Streamlit-native so we can react)
    col_kill, col_refresh = st.columns([1, 1])
    with col_kill:
        confirm_key = "kill_confirm"
        if st.session_state.get(confirm_key):
            if st.button("⚠️ Really kill bridge?", type="primary", use_container_width=True, key="kill_go"):
                ok, msg = kill_bridge()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.session_state[confirm_key] = False
        else:
            if st.button("🛑 Kill bridge", use_container_width=True, key="kill_req"):
                st.session_state[confirm_key] = True
                st.rerun()
    with col_refresh:
        if st.button("🔄 Refresh now", use_container_width=True, key="refresh_now"):
            st.rerun()


def render_outcomes_donut(records: List[ExecutionRecord]) -> None:
    if not records:
        st.info("No executions completed yet today.")
        return
    counter = Counter(r.status for r in records)
    df = pd.DataFrame(
        {"status": list(counter.keys()), "count": list(counter.values())}
    )
    color_scale = alt.Scale(
        domain=["passed", "failed", "unknown"],
        range=["#4ade80", "#f87171", "#9ca3af"],
    )
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=55, outerRadius=95, stroke="#0e1117", strokeWidth=2)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("status:N", scale=color_scale, legend=alt.Legend(orient="right")),
            tooltip=["status", "count"],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)


def render_recent_history(records: List[ExecutionRecord]) -> None:
    if not records:
        st.info("No executions today yet.")
        return
    recent = list(reversed(records))[:12]
    rows = []
    for r in recent:
        badge = "✅" if r.status == "passed" else "❌"
        rows.append(
            {
                "": badge,
                "Time": r.ts.strftime("%H:%M:%S"),
                "Run": r.run_number or "—",
                "Test case": r.test_case[:48],
                "Duration": human_duration(r.duration_sec),
                "Reason": r.failure_reason or "",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(420, 48 + 36 * len(df)))


def render_failures_breakdown(records: List[ExecutionRecord]) -> None:
    failures = [r for r in records if r.status == "failed"]
    if not failures:
        st.success("No failures today — clean run so far.")
        return
    counter = Counter(r.failure_reason or "unknown" for r in failures)
    df = pd.DataFrame(
        {"reason": list(counter.keys()), "count": list(counter.values())}
    ).sort_values("count", ascending=True)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("count:Q", title=None),
            y=alt.Y("reason:N", sort="-x", title=None),
            color=alt.Color(
                "count:Q",
                scale=alt.Scale(scheme="reds"),
                legend=None,
            ),
            tooltip=["reason", "count"],
        )
        .properties(height=max(180, 44 * len(df)))
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander(f"🔍 See all {len(failures)} failures", expanded=False):
        rows = [
            {
                "Time": r.ts.strftime("%H:%M:%S"),
                "Run": r.run_number or "—",
                "Test case": r.test_case[:60],
                "Reason": r.failure_reason or "unknown",
                "Duration": human_duration(r.duration_sec),
            }
            for r in reversed(failures)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_log_tail() -> None:
    service_names = list(SERVICES.keys()) + ["bridge"]
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        svc = st.selectbox("Service", service_names, index=service_names.index("execute"))
    with col2:
        flt = st.text_input("Filter (regex, case-insensitive)", value="", placeholder="e.g. ERROR|TURN")
    with col3:
        n = st.slider("Lines", min_value=50, max_value=MAX_TAIL_LINES, value=TAIL_LINES_DEFAULT, step=50)

    if svc == "bridge":
        base = BRIDGE_LOG_BASENAME
    else:
        base = SERVICES[svc][1]
    path = today_log(base)

    if not path.exists():
        st.warning(f"Log file not found yet: `{path.name}`")
        return

    lines = tail_file(path, n=n)
    if flt:
        try:
            pat = re.compile(flt, re.IGNORECASE)
            lines = [l for l in lines if pat.search(l)]
        except re.error as e:
            st.error(f"Bad regex: {e}")

    st.caption(f"`{path.name}` — {len(lines)} lines shown")

    def _color(line: str) -> str:
        lower = line.lower()
        if "error" in lower or "traceback" in lower or "❌" in line or "🚨" in line:
            cls = "log-err"
        elif "warning" in lower or "⚠️" in line:
            cls = "log-warn"
        elif "✓" in line or "✅" in line or "completed successfully" in lower:
            cls = "log-ok"
        else:
            cls = "log-info"
        return f'<span class="{cls}">{_escape(line)}</span>'

    body = "<br>".join(_color(l) for l in lines)
    # Single-line HTML — no markdown indent trap possible.
    st.markdown(f'<div class="log-box">{body}</div>', unsafe_allow_html=True)


def render_services_tab(svc_state: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for name, info in svc_state.items():
        rows.append(
            {
                "": "🟢" if info["alive"] else "🔴",
                "Service": name,
                "PID": info.get("pid") or "—",
                "Uptime": info.get("etime") or "—",
                "Status": "alive" if info["alive"] else "down",
            }
        )
    bridge = bridge_process()
    if bridge:
        rows.append(
            {
                "": "🔵",
                "Service": "bridge (spawned)",
                "PID": bridge["pid"],
                "Uptime": bridge["etime"],
                "Status": "running test",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Log files on disk")
    log_rows = []
    if LOG_DIR.exists():
        for p in sorted(LOG_DIR.glob(f"*_{datetime.now().strftime('%Y%m%d')}.log")):
            try:
                size = p.stat().st_size
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
            except Exception:
                continue
            log_rows.append({"File": p.name, "Size": human_bytes(size), "Last write": mtime})
    if log_rows:
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No log files for today found yet.")


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    # ---- Auto-refresh control (sidebar) -------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Refresh")
        auto_on = st.toggle("Auto-refresh", value=True, key="auto_on")
        interval_sec = st.slider(
            "Every N seconds",
            min_value=2,
            max_value=30,
            value=int(AUTO_REFRESH_MS / 1000),
            step=1,
            key="refresh_interval",
        )
        if st.button("🔄 Refresh now", use_container_width=True, key="refresh_sidebar"):
            st.rerun()
        if not HAS_AUTOREFRESH:
            st.caption(
                "ℹ️ `streamlit-autorefresh` not installed. Using full-page "
                "reload fallback — install it for smoother updates: "
                "`pip install streamlit-autorefresh`"
            )
        st.caption(f"Last tick: {datetime.now().strftime('%H:%M:%S')}")

    # ---- Trigger the auto-refresh -------------------------------------------
    if auto_on:
        if HAS_AUTOREFRESH:
            # Clean server-side rerun every N seconds. Streamlit re-executes
            # the script from the top, so all data_layer functions re-query
            # the logs/filesystem and re-render. No flicker, no scroll reset.
            st_autorefresh(interval=interval_sec * 1000, key="dashboard_tick")
        else:
            # Fallback: full page reload via an iframe component. Components
            # ARE allowed to execute JS (unlike st.markdown). `window.parent`
            # targets the outer Streamlit frame, not the component iframe.
            components.html(
                f"""
<script>
  (function() {{
    setTimeout(function() {{
      try {{ window.parent.location.reload(); }}
      catch (e) {{ window.location.reload(); }}
    }}, {interval_sec * 1000});
  }})();
</script>
""",
                height=0,
            )

    svc = service_state()
    render_hero(svc)
    render_metric_strip(svc)

    tab_live, tab_logs, tab_failures, tab_services = st.tabs(
        ["🚀 Live", "📝 Logs", "⚠️ Failures", "⚙️ Services"]
    )

    records = parse_recent_executions()

    with tab_live:
        col_a, col_b = st.columns([5, 7], gap="large")
        with col_a:
            bridge_status = parse_bridge_status()
            render_bridge_card(bridge_status)
        with col_b:
            st.markdown("### 🎯 Today's outcomes")
            render_outcomes_donut(records)
            st.markdown("### 📜 Recent executions")
            render_recent_history(records)

    with tab_logs:
        render_log_tail()

    with tab_failures:
        st.markdown("### Failure breakdown (today)")
        render_failures_breakdown(records)

    with tab_services:
        render_services_tab(svc)


if __name__ == "__main__":
    main()
