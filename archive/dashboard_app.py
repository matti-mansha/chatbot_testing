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

# Reuse the KPI-computation machinery from calculate_kpis.py instead of
# reimplementing anything. All the calc_* functions are top-level and
# importable. This gives us a single source of truth for KPI math so
# the dashboard view and the CLI report stay in sync.
try:
    import calculate_kpis as kpi_calc  # noqa: E402
    HAS_KPI_CALC = True
except Exception:
    HAS_KPI_CALC = False


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
    """Return info about the first running playwright bridge subprocess (legacy)."""
    bridges = bridge_processes()
    return bridges[0] if bridges else None


def bridge_processes() -> List[Dict[str, str]]:
    """Return ALL currently-running playwright bridge subprocesses (one per
    parallel worker in the execution pool)."""
    found: List[Dict[str, str]] = []
    for p in ps_processes():
        if "playwright_bridge_bot_headless.py" in p["cmd"] and "grep" not in p["cmd"]:
            found.append(p)
    return found


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
    # One pill per running bridge (there may be multiple in parallel mode)
    bridge_pill = ""
    bridges = bridge_processes()
    for b in bridges:
        bridge_pill += (
            f'<div class="status-pill" style="background: rgba(96,165,250,0.12); border-color: rgba(96,165,250,0.3);">'
            f'<span class="status-dot dot-alive" style="background:#60a5fa;"></span>'
            f'bridge {b["pid"]}<span class="muted">· {b["etime"]}</span></div>'
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


def render_bridge_cards() -> None:
    """
    Render one card per running bridge process (parallel mode).
    Falls back to a single "pipeline idle" card if nothing is running.
    """
    bridges = bridge_processes()
    if not bridges:
        html("""
            <div class="card">
            <div class="card-title">CURRENT BRIDGE</div>
            <div class="big-name">💤 Pipeline idle</div>
            <div class="muted">No Playwright bridge subprocess is running right now.
            The next run will fire when prepare/execute sees a pending execution.</div>
            </div>
        """)
        return

    # Parse the shared bridge log once and distribute the findings to
    # each running bridge by PID / test-case correlation. We use a simple
    # heuristic: the most recent TURN marker before each bridge's
    # "STARTING TEST EXECUTION" line belongs to its previous attempt and
    # can be ignored; everything after is "live".
    per_pid = _parse_bridge_log_per_process(bridges)

    # Render a card per bridge
    for b in bridges:
        pid = b["pid"]
        state = per_pid.get(pid) or BridgeStatus()
        state.running = True
        if not state.test_case:
            # Fall back to argv from ps cmdline
            cmd = b.get("cmd", "")
            m = re.search(r"playwright_bridge_bot_headless\.py\s+(.*)$", cmd)
            if m:
                state.test_case = m.group(1).split(" ", 1)[0][:80]
        if state.elapsed_sec <= 0:
            state.elapsed_sec = float(parse_etime_to_seconds(b.get("etime", "")))
        _render_single_bridge_card(state, pid=pid)

    # Pool-wide kill button row
    col_kill, col_refresh = st.columns([1, 1])
    with col_kill:
        confirm_key = "kill_all_confirm"
        if st.session_state.get(confirm_key):
            if st.button(
                f"⚠️ Kill ALL {len(bridges)} bridges?",
                type="primary",
                use_container_width=True,
                key="kill_all_go",
            ):
                killed = 0
                for b in bridges:
                    try:
                        os.kill(int(b["pid"]), signal.SIGKILL)
                        killed += 1
                    except Exception:
                        pass
                st.success(f"SIGKILL sent to {killed}/{len(bridges)} bridges")
                st.session_state[confirm_key] = False
        else:
            if st.button(
                f"🛑 Kill all {len(bridges)} bridges",
                use_container_width=True,
                key="kill_all_req",
            ):
                st.session_state[confirm_key] = True
                st.rerun()
    with col_refresh:
        if st.button("🔄 Refresh now", use_container_width=True, key="refresh_now"):
            st.rerun()


def _parse_bridge_log_per_process(bridges: List[Dict[str, str]]) -> Dict[str, "BridgeStatus"]:
    """
    Read the shared playwright_bridge_<date>.log and split the latest
    activity into a BridgeStatus per running bridge process.

    Strategy: scan the log line-by-line and maintain a "current" state.
    Every time we encounter a `STARTING TEST EXECUTION (Attempt 1/...)` we
    rotate into a new BridgeStatus. When we're done, return the last few
    BridgeStatus objects — one per running bridge, in order of latest to
    oldest — matched to process PIDs by list position (the most recently
    spawned PID maps to the most recent StartingTestExecution, and so on).

    This is a heuristic but works well in practice because:
      * Each bridge logs its STARTING line when it boots
      * Bridges finish in roughly the same order they started
      * We can tie-break by elapsed seconds from `ps etime`

    If the heuristic fails (log lines interleaved across PIDs), we fall
    back to "most recent state for all bridges" which is still better
    than nothing.
    """
    result: Dict[str, BridgeStatus] = {}
    if not bridges:
        return result

    path = today_log(BRIDGE_LOG_BASENAME)
    lines = tail_file(path, n=2000)
    if not lines:
        # No log yet: give every bridge an empty status so the UI still
        # renders a card.
        for b in bridges:
            s = BridgeStatus()
            s.running = True
            result[b["pid"]] = s
        return result

    # Collect "segments" — each segment is a run of lines between two
    # `STARTING TEST EXECUTION (Attempt 1/` markers (i.e., a new test,
    # not a retry of the same test).
    segments: List[BridgeStatus] = []
    current = BridgeStatus()
    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        msg = parsed["msg"]
        try:
            ts = datetime.strptime(parsed["ts"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = None

        start_match = re.search(
            r"STARTING TEST EXECUTION\s*\(Attempt 1/(\d+)", msg
        )
        if start_match:
            # New segment = new test case spawn
            if current.started_at or current.test_case:
                segments.append(current)
            current = BridgeStatus()
            current.attempt = 1
            current.attempt_total = int(start_match.group(1))
            current.started_at = ts
            current.last_event_at = ts
            continue

        if ts is not None:
            current.last_event_at = ts

        m = re.search(r"TURN (\d+)/(\d+)", msg)
        if m:
            current.turn = int(m.group(1))
            current.max_turns = int(m.group(2))

        if msg.startswith("Content:"):
            current.last_mila_reply = msg[len("Content:"):].strip()

        m = re.search(
            r"(?:RE)?STARTING TEST EXECUTION\s*\(Attempt (\d+)/(\d+)"
            r"(?:, general=(\d+)/\d+, mila-reject=(\d+)/\d+)?",
            msg,
        )
        if m:
            current.attempt = int(m.group(1))
            current.attempt_total = int(m.group(2))
            if m.group(3):
                current.general_restart_attempt = int(m.group(3))
            if m.group(4):
                current.mila_reject_attempt = int(m.group(4))

        m = re.search(r"Test case:\s+(.+)$", msg)
        if m and not current.test_case:
            current.test_case = m.group(1).strip()

    # Flush final segment
    if current.started_at or current.test_case:
        segments.append(current)

    # Match the last N segments to the N bridges, newest-first. If we have
    # fewer segments than bridges, pad with empty state objects.
    tail = segments[-len(bridges):] if segments else []
    # Bridges in ps order (usually ordered by PID / start time). Sort by
    # elapsed time descending (longest-running first) to align with log
    # segments (earliest started first).
    bridges_sorted = sorted(
        bridges,
        key=lambda b: parse_etime_to_seconds(b.get("etime", "")),
        reverse=True,
    )

    for i, b in enumerate(bridges_sorted):
        if i < len(tail):
            result[b["pid"]] = tail[-(i + 1)]  # reverse order: oldest segment to oldest bridge
        else:
            result[b["pid"]] = BridgeStatus()
            result[b["pid"]].running = True

    # Cross-reference with execute log for run number on the MOST RECENT one
    run_num = _latest_run_number_from_execute_log()
    if run_num:
        # Best-effort: attach to the newest bridge (shortest etime)
        newest = min(bridges, key=lambda b: parse_etime_to_seconds(b.get("etime", "")))
        if newest["pid"] in result and not result[newest["pid"]].run_number:
            result[newest["pid"]].run_number = run_num

    return result


def _render_single_bridge_card(status: "BridgeStatus", pid: str) -> None:
    """Render a single bridge card (called once per parallel worker)."""

    pct = 0
    if status.max_turns:
        pct = min(100, int(100 * status.turn / status.max_turns))
    elapsed = human_duration(status.elapsed_sec)
    test_case = _escape(status.test_case or "(unknown)")
    persona = _escape(status.persona or "")
    last_mila = _escape(status.last_mila_reply)[:400]

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
        <div class="card-title">▶ BRIDGE · PID {pid}</div>
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

    # Per-bridge kill button (keyed by PID so multiple cards don't collide)
    confirm_key = f"kill_{pid}_confirm"
    if st.session_state.get(confirm_key):
        if st.button(
            f"⚠️ Really kill PID {pid}?",
            type="primary",
            use_container_width=True,
            key=f"kill_{pid}_go",
        ):
            try:
                os.kill(int(pid), signal.SIGKILL)
                st.success(f"SIGKILL sent to PID {pid}")
            except Exception as e:
                st.error(f"Kill failed: {e}")
            st.session_state[confirm_key] = False
    else:
        if st.button(
            f"🛑 Kill PID {pid}",
            use_container_width=True,
            key=f"kill_{pid}_req",
        ):
            st.session_state[confirm_key] = True
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


# =============================================================================
# KPI TAB
# =============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def _kpi_test_runs() -> List[Tuple[str, str, float, int, int, Optional[str]]]:
    """
    Return a list of test runs that have at least one evaluated execution.
    Each tuple: (run_prefix, run_title, avg_score_or_0, num_evals, num_total, status).
    The dashboard uses this for the run selector.

    Cached for 60s so the selector doesn't re-query Notion on every tick.
    """
    if not HAS_KPI_CALC:
        return []
    try:
        test_runs_db_id = kpi_calc.find_database_in_page(kpi_calc.TEST_RUNS_PAGE_ID)
        if not test_runs_db_id:
            return []
        rows = kpi_calc.notion_query_all(test_runs_db_id)
    except Exception:
        return []

    result: List[Tuple[str, str, float, int, int, Optional[str]]] = []
    for row in rows:
        props = row.get("properties", {})
        # "Test Run Number" is the title property
        title = ""
        for p in props.values():
            if p.get("type") == "title":
                title = "".join(c.get("plain_text", "") for c in p.get("title", []))
                break
        if not title:
            continue
        score = kpi_calc.extract_property_value(row, "Score") or 0
        num_evals = int(kpi_calc.extract_property_value(row, "Number of evaluations") or 0)
        num_total = int(kpi_calc.extract_property_value(row, "Total number of test cases") or 0)
        status = kpi_calc.extract_property_value(row, "Status")
        result.append((title, title, float(score), num_evals, num_total, status))

    # Sort newest-ish first (by title descending works for "TR1", "TR2", ...)
    result.sort(key=lambda t: t[0], reverse=True)
    return result


@st.cache_data(ttl=60, show_spinner="Fetching KPIs from Notion…")
def _kpi_payload_for_run(run_prefix: str) -> Optional[Dict[str, Any]]:
    """
    Fetch + aggregate all KPIs for the given test run, reusing
    calculate_kpis.py's aggregation functions. Cached for 60s so a 28-test
    run's 28 Notion page fetches only happen once a minute.

    Returns a dict with:
      * raw_data: list of per-execution dicts (with detail KPIs)
      * kpis: dict of all aggregate KPIs (pass_rate, avg_score, etc.)
      * num_evaluated: int
    Or None if nothing to show.
    """
    if not HAS_KPI_CALC:
        return None
    try:
        test_exec_db_id = kpi_calc.find_database_in_page(
            kpi_calc.TEST_CASE_EXECUTIONS_PAGE_ID
        )
        if not test_exec_db_id:
            return None
        executions = kpi_calc.get_evaluated_executions(test_exec_db_id, run_prefix)
        if not executions:
            return None
        data = kpi_calc.collect_execution_data(executions, fetch_kpi_details=True)
        kpis = {
            "pass_rate":             kpi_calc.calc_pass_rate(data),
            "avg_score":             kpi_calc.calc_avg_score(data),
            "score_distribution":    kpi_calc.calc_score_distribution(data),
            "weakest_kpi":           kpi_calc.calc_weakest_kpi(data),
            "safety_compliance":     kpi_calc.calc_safety_compliance(data),
            "workflow_compliance":   kpi_calc.calc_workflow_compliance(data),
            "avg_turns":             kpi_calc.calc_avg_turns(data),
            "avg_duration":          kpi_calc.calc_avg_duration(data),
            "overstaying_rate":      kpi_calc.calc_overstaying_rate(data),
            "completeness_velocity": kpi_calc.calc_completeness_velocity(data),
            "premature_ending_rate": kpi_calc.calc_premature_ending_rate(data),
            "persona_fairness":      kpi_calc.calc_persona_fairness(data),
            "worst_test_cases":      kpi_calc.calc_worst_test_cases(data),
            "best_test_cases":       kpi_calc.calc_best_test_cases(data),
        }
        return {
            "run_prefix": run_prefix,
            "num_evaluated": len(data),
            "raw_data": data,
            "kpis": kpis,
        }
    except Exception as e:
        st.error(f"Failed to fetch KPIs for {run_prefix}: {e}")
        return None


def _score_color(score: Optional[float]) -> str:
    if score is None:
        return "#6b7280"
    if score >= 80:
        return "#4ade80"
    if score >= 60:
        return "#facc15"
    if score >= 40:
        return "#fb923c"
    return "#f87171"


def render_kpis_tab() -> None:
    """
    Render the full 14-KPI dashboard, organized by the same taxonomy as
    calculate_kpis.py (Quality / Efficiency / Consistency / Topic-Level):

        Quality (6):
            1. Overall Pass Rate
            2. Average Evaluation Score
            3. Score Distribution
            4. Weakest KPI per Run (+ 9 rubric dimensions chart)
            5. Safety Compliance Rate
            6. Support Workflow Compliance Rate
        Efficiency (5):
            7. Avg Turns to Resolution
            8. Avg Conversation Duration
            9. Overstaying Rate
            10. Completeness Velocity
            11. Premature Ending Rate
        Consistency (1):
            12. Persona Fairness Gap
        Topic-Level (2):
            13. Worst 5 Test Cases
            14. Best 5 Test Cases
    """
    if not HAS_KPI_CALC:
        st.error(
            "KPI computation module (calculate_kpis.py) could not be imported. "
            "The KPI tab is unavailable."
        )
        return

    runs = _kpi_test_runs()
    if not runs:
        st.info(
            "No test runs with evaluated executions found yet. Trigger a run "
            "from Notion and wait for the evaluator to produce results."
        )
        return

    # --- Run selector -------------------------------------------------------
    run_labels = [
        f"{r[0]} · {r[3]}/{r[4]} evals · avg {r[2]:.1f}/100"
        for r in runs
    ]
    sel_col, btn_col = st.columns([4, 1])
    with sel_col:
        sel_idx = st.selectbox(
            "Test run",
            options=list(range(len(runs))),
            format_func=lambda i: run_labels[i],
            index=0,
            key="kpi_run_select",
        )
    with btn_col:
        if st.button("🔄 Refresh KPIs", use_container_width=True, key="kpi_refresh"):
            _kpi_payload_for_run.clear()
            _kpi_test_runs.clear()
            st.rerun()

    run_prefix = runs[sel_idx][0]
    run_status = runs[sel_idx][5] or "—"
    num_total = runs[sel_idx][4]

    # --- Analytics export callout (for the downstream analytics partner) ---
    # Show a compact row with direct download links for today's and
    # yesterday's JSONL event streams. Both URLs are served by nginx at
    # /analytics/ behind the same Basic Auth as the dashboard itself.
    _today = datetime.now(timezone.utc).strftime("%Y%m%d")
    _yday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    exp_col1, exp_col2, exp_col3 = st.columns([2, 2, 3])
    with exp_col1:
        st.link_button(
            f"📥 Today's events ({_today})",
            f"/analytics/events_{_today}.txt",
            use_container_width=True,
        )
    with exp_col2:
        st.link_button(
            f"📥 Yesterday ({_yday})",
            f"/analytics/events_{_yday}.txt",
            use_container_width=True,
        )
    with exp_col3:
        st.link_button(
            "📂 Browse all analytics",
            "/analytics/",
            use_container_width=True,
        )
    st.caption(
        "📊 Analytics event stream — `[[...]]` wrapped JSON per the "
        "project chat-metadata format. Schema: `docs/ANALYTICS_SCHEMA.md`. "
        "Share these links (with Basic Auth credentials) with your analytics partner."
    )

    payload = _kpi_payload_for_run(run_prefix)
    if not payload:
        st.warning(f"No evaluated executions for {run_prefix} yet.")
        return

    kpis = payload["kpis"]
    data = payload["raw_data"]
    num_evaluated = payload["num_evaluated"]
    progress_pct = (num_evaluated / num_total * 100) if num_total else 0.0

    # =======================================================================
    # Scorecard strip (headline metrics)
    # =======================================================================
    avg_score_info = kpis.get("avg_score", {}) or {}
    avg_score = avg_score_info.get("value")
    pass_rate_info = kpis.get("pass_rate", {}) or {}
    pass_rate = pass_rate_info.get("value")
    weakest = kpis.get("weakest_kpi", {}) or {}
    weakest_name = (weakest.get("value") or "").replace("_", " ").title() if weakest.get("value") else "—"
    weakest_score = weakest.get("score")

    avg_score_color = _score_color(avg_score)
    pass_rate_color = _score_color(pass_rate)

    pass_detail = pass_rate_info.get("detail", "—")
    avg_detail = avg_score_info.get("detail", "—")

    html(f"""
        <div class="metric-strip">
        <div class="metric-cell">
        <div class="metric-label">Avg score</div>
        <div class="metric-value" style="color:{avg_score_color};">{avg_score if avg_score is not None else '—'}</div>
        <div class="metric-sub">{_escape(avg_detail)}</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">Pass rate</div>
        <div class="metric-value" style="color:{pass_rate_color};">{str(pass_rate) + '%' if pass_rate is not None else '—'}</div>
        <div class="metric-sub">{_escape(pass_detail)}</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">Progress</div>
        <div class="metric-value">{num_evaluated}<span style="color:#6b7280; font-size:16px;"> / {num_total}</span></div>
        <div class="metric-sub">{progress_pct:.0f}% evaluated · status: {_escape(run_status)}</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">Weakest KPI</div>
        <div class="metric-value" style="font-size:16px; color:#f87171;">{_escape(weakest_name)}</div>
        <div class="metric-sub">{f'{weakest_score:.1f}/100' if weakest_score is not None else '—'}</div>
        </div>
        </div>
    """)

    # =======================================================================
    # A) QUALITY KPIs (6)
    # =======================================================================
    st.markdown("## 📊 A) Quality KPIs (6)")
    st.caption(
        "1. Pass Rate · 2. Avg Score · 3. Score Distribution · "
        "4. Weakest KPI · 5. Safety Compliance · 6. Workflow Compliance"
    )

    col_bar, col_donut = st.columns([7, 5], gap="large")

    with col_bar:
        st.markdown("#### 4. Weakest KPI per Run — 9-dimension rubric breakdown")
        kpi_scores_all: Dict[str, List[int]] = {}
        for d in data:
            for kpi_name, score in (d.get("kpis") or {}).items():
                if score is not None:
                    kpi_scores_all.setdefault(kpi_name, []).append(int(score))
        if kpi_scores_all:
            kpi_rows = []
            for kpi_name in kpi_calc.KPI_NAMES:
                scores = kpi_scores_all.get(kpi_name, [])
                avg = sum(scores) / len(scores) if scores else 0.0
                kpi_rows.append(
                    {
                        "kpi": kpi_name.replace("_", " ").title(),
                        "avg": round(avg, 1),
                        "n": len(scores),
                    }
                )
            df = pd.DataFrame(kpi_rows).sort_values("avg", ascending=True)
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadius=4, height=22)
                .encode(
                    x=alt.X("avg:Q", title="Average score", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("kpi:N", sort="-x", title=None),
                    color=alt.Color(
                        "avg:Q",
                        scale=alt.Scale(
                            domain=[0, 40, 60, 80, 100],
                            range=["#f87171", "#fb923c", "#facc15", "#a3e635", "#4ade80"],
                        ),
                        legend=None,
                    ),
                    tooltip=["kpi", "avg", "n"],
                )
                .properties(height=320)
            )
            text = (
                alt.Chart(df)
                .mark_text(
                    align="left", baseline="middle", dx=4, fontSize=11, color="#f3f4f6"
                )
                .encode(x="avg:Q", y=alt.Y("kpi:N", sort="-x"), text="avg:Q")
            )
            st.altair_chart(chart + text, use_container_width=True)
        else:
            st.info("No detailed KPI data available. Evaluator hasn't produced results yet.")

    with col_donut:
        st.markdown("#### 3. Score Distribution")
        dist_info = kpis.get("score_distribution", {}) or {}
        dist_val = dist_info.get("value")
        if dist_val and isinstance(dist_val, dict):
            band_order = [
                "Excellent (81-100)",
                "Good (61-80)",
                "Acceptable (41-60)",
                "Poor (1-40)",
            ]
            band_colors = ["#4ade80", "#a3e635", "#facc15", "#f87171"]
            band_rows = []
            for band in band_order:
                val_str = dist_val.get(band, "0/0 (0%)")
                m = re.match(r"(\d+)/", str(val_str))
                count = int(m.group(1)) if m else 0
                band_rows.append({"band": band, "count": count})
            band_df = pd.DataFrame(band_rows)
            if band_df["count"].sum() > 0:
                donut = (
                    alt.Chart(band_df)
                    .mark_arc(
                        innerRadius=55, outerRadius=95, stroke="#0e1117", strokeWidth=2
                    )
                    .encode(
                        theta=alt.Theta("count:Q", stack=True),
                        color=alt.Color(
                            "band:N",
                            scale=alt.Scale(domain=band_order, range=band_colors),
                            legend=alt.Legend(orient="right", title=None),
                            sort=band_order,
                        ),
                        tooltip=["band", "count"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(donut, use_container_width=True)
            else:
                st.info("No scores yet.")
        else:
            st.info("No score distribution data.")

    # --- KPIs 5 & 6: Safety + Workflow compliance (specialized rates) ------
    safety_info = kpis.get("safety_compliance", {}) or {}
    workflow_info = kpis.get("workflow_compliance", {}) or {}
    safety_val = safety_info.get("value")
    workflow_val = workflow_info.get("value")
    safety_detail = safety_info.get("detail", "—")
    workflow_detail = workflow_info.get("detail", "—")

    def _fmt_pct_or_na(v: Any) -> str:
        if v is None or v == "N/A":
            return "N/A"
        return f"{v}%"

    safety_color = _score_color(safety_val if isinstance(safety_val, (int, float)) else None)
    workflow_color = _score_color(workflow_val if isinstance(workflow_val, (int, float)) else None)

    html(f"""
        <div class="metric-strip" style="grid-template-columns: repeat(2, 1fr);">
        <div class="metric-cell">
        <div class="metric-label">5. Safety Compliance Rate</div>
        <div class="metric-value" style="color:{safety_color};">{_fmt_pct_or_na(safety_val)}</div>
        <div class="metric-sub">{_escape(safety_detail)}</div>
        </div>
        <div class="metric-cell">
        <div class="metric-label">6. Support Workflow Compliance Rate</div>
        <div class="metric-value" style="color:{workflow_color};">{_fmt_pct_or_na(workflow_val)}</div>
        <div class="metric-sub">{_escape(workflow_detail)}</div>
        </div>
        </div>
    """)

    # =======================================================================
    # B) EFFICIENCY KPIs (5)
    # =======================================================================
    st.markdown("## ⚡ B) Efficiency KPIs (5)")
    st.caption(
        "7. Avg Turns · 8. Avg Duration · 9. Overstaying Rate · "
        "10. Completeness Velocity · 11. Premature Ending Rate"
    )

    eff_cols = st.columns(5)
    with eff_cols[0]:
        v = kpis.get("avg_turns", {}) or {}
        st.metric("7. Avg turns", f"{v.get('value','—')}", help=v.get("detail", ""))
    with eff_cols[1]:
        v = kpis.get("avg_duration", {}) or {}
        val = v.get("value")
        disp = human_duration(val) if val is not None else "—"
        st.metric("8. Avg duration", disp, help=v.get("detail", ""))
    with eff_cols[2]:
        v = kpis.get("overstaying_rate", {}) or {}
        val = v.get("value")
        st.metric(
            "9. Overstaying rate",
            f"{val}%" if val is not None else "—",
            help=v.get("detail", ""),
        )
    with eff_cols[3]:
        v = kpis.get("completeness_velocity", {}) or {}
        val = v.get("value")
        if isinstance(val, (int, float)):
            disp = f"{val:+.1f} pts/turn"
        else:
            disp = "N/A"
        st.metric(
            "10. Completeness velocity",
            disp,
            help=v.get("detail", ""),
        )
    with eff_cols[4]:
        v = kpis.get("premature_ending_rate", {}) or {}
        val = v.get("value")
        st.metric(
            "11. Premature endings",
            f"{val}%" if val is not None else "—",
            help=v.get("detail", ""),
        )

    # =======================================================================
    # C) CONSISTENCY KPI (1) — Persona Fairness Gap
    # =======================================================================
    st.markdown("## 🔄 C) Consistency KPI (1)")
    st.caption("12. Persona Fairness Gap")
    pf = kpis.get("persona_fairness", {}) or {}
    pf_val = pf.get("value")
    pf_groups = pf.get("groups") or {}
    if isinstance(pf_val, (int, float)):
        if pf_val > 10:
            st.warning(
                f"⚖️ **12. Persona fairness gap: {pf_val} points** — "
                f"{pf.get('detail', '')}. MILA scores noticeably different across "
                f"persona groups — possible bias worth investigating."
            )
        else:
            st.success(
                f"⚖️ **12. Persona fairness gap: {pf_val} points** — "
                f"{pf.get('detail', '')} — no significant bias detected."
            )
        # Bar chart of persona group averages
        if pf_groups:
            pf_df = pd.DataFrame(
                [{"group": g, "avg": s} for g, s in pf_groups.items()]
            )
            pf_chart = (
                alt.Chart(pf_df)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("avg:Q", title="Avg score", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("group:N", sort="-x", title=None),
                    color=alt.Color(
                        "avg:Q",
                        scale=alt.Scale(
                            domain=[0, 40, 60, 80, 100],
                            range=["#f87171", "#fb923c", "#facc15", "#a3e635", "#4ade80"],
                        ),
                        legend=None,
                    ),
                    tooltip=["group", "avg"],
                )
                .properties(height=110)
            )
            st.altair_chart(pf_chart, use_container_width=True)
    else:
        st.info("12. Persona fairness gap — need at least 2 persona groups to compare.")

    # =======================================================================
    # D) TOPIC-LEVEL KPIs (2) — Worst + Best 5 test cases
    # =======================================================================
    st.markdown("## 🎯 D) Topic-Level KPIs (2)")
    st.caption("13. Worst 5 Test Cases · 14. Best 5 Test Cases")

    worst_info = kpis.get("worst_test_cases", {}) or {}
    best_info = kpis.get("best_test_cases", {}) or {}
    worst_tcs = worst_info.get("value") or []
    best_tcs = best_info.get("value") or []

    col_worst, col_best = st.columns(2, gap="large")
    with col_worst:
        st.markdown("#### 13. Worst 5 (priority fix list)")
        if worst_tcs:
            worst_df = pd.DataFrame(
                [{"Score": s, "Test case": n[:60]} for n, s in worst_tcs]
            )
            st.dataframe(
                worst_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", format="%.0f", min_value=0, max_value=100
                    ),
                },
            )
        else:
            st.info("No data yet.")

    with col_best:
        st.markdown("#### 14. Best 5 (strengths)")
        if best_tcs:
            best_df = pd.DataFrame(
                [{"Score": s, "Test case": n[:60]} for n, s in best_tcs]
            )
            st.dataframe(
                best_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", format="%.0f", min_value=0, max_value=100
                    ),
                },
            )
        else:
            st.info("No data yet.")

    # =======================================================================
    # Full per-test-case leaderboard (for deeper inspection)
    # =======================================================================
    with st.expander("📋 All evaluated executions (sortable)", expanded=False):
        rows = []
        for d in data:
            score = d.get("overall_score")
            result = d.get("result") or "—"
            rows.append(
                {
                    "Score": round(score, 1) if score is not None else None,
                    "Result": result,
                    "Run": d.get("run_number", ""),
                    "Test case": (d.get("test_case_name") or "")[:60],
                    "Persona": (d.get("persona") or "")[:40],
                    "Turns": d.get("num_turns"),
                    "Duration": human_duration(d.get("duration")) if d.get("duration") else "—",
                }
            )
        if rows:
            leaderboard = pd.DataFrame(rows).sort_values(
                "Score", ascending=True, na_position="last"
            )
            st.dataframe(
                leaderboard,
                use_container_width=True,
                hide_index=True,
                height=min(600, 50 + 36 * len(leaderboard)),
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        format="%.0f",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )
        else:
            st.info("No completed evaluations yet.")

    # =======================================================================
    # Historical trend across test runs
    # =======================================================================
    if len(runs) > 1:
        st.markdown("## 📈 Historical trend (avg score across test runs)")
        hist_df = pd.DataFrame(
            [
                {"run": r[0], "avg_score": r[2], "evals": r[3]}
                for r in runs
                if r[2] > 0
            ]
        )
        if not hist_df.empty:
            line = (
                alt.Chart(hist_df)
                .mark_line(point=True, strokeWidth=3, color="#60a5fa")
                .encode(
                    x=alt.X("run:N", title="Test run", sort=None),
                    y=alt.Y("avg_score:Q", title="Avg score", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["run", "avg_score", "evals"],
                )
                .properties(height=220)
            )
            st.altair_chart(line, use_container_width=True)


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

    tab_live, tab_kpis, tab_logs, tab_failures, tab_services = st.tabs(
        ["🚀 Live", "📊 KPIs", "📝 Logs", "⚠️ Failures", "⚙️ Services"]
    )

    records = parse_recent_executions()

    with tab_live:
        col_a, col_b = st.columns([5, 7], gap="large")
        with col_a:
            render_bridge_cards()
        with col_b:
            st.markdown("### 🎯 Today's outcomes")
            render_outcomes_donut(records)
            st.markdown("### 📜 Recent executions")
            render_recent_history(records)

    with tab_kpis:
        render_kpis_tab()

    with tab_logs:
        render_log_tail()

    with tab_failures:
        st.markdown("### Failure breakdown (today)")
        render_failures_breakdown(records)

    with tab_services:
        render_services_tab(svc)


if __name__ == "__main__":
    main()
