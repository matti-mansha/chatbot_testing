#!/usr/bin/env python3
"""
Log retention sweeper.

Keeps logs/ and diagnostics/ from growing unbounded (observed: 813 MB logs,
9.4 GB diagnostics in a single day of runs).

Policy (all configurable via env vars, defaults baked in):

    logs/ — per-day structured .log files
        * KEEP_LOG_DAYS=14      — today and the previous N-1 days are kept
                                   uncompressed so they're live-tailable
        * GZIP_LOG_DAYS=3       — files older than this get gzipped in place
        * DELETE_LOG_DAYS=30    — gzipped files older than this get removed
        * Rotated files (foo.log.1, foo.log.2, ...) are treated as day-based
          peers of the main file for the purpose of rotation.

    diagnostics/ — session-indexed JSON + DOM snapshots from the bridge
        * KEEP_DIAG_DAYS=3      — diagnostics older than N days are deleted
                                   outright (they're only useful right after
                                   a failure to debug why)

    screenshots/
        * KEEP_SCREENSHOT_DAYS=3

Run directly:
    env/bin/python log_retention.py            # normal run, writes to stdout
    env/bin/python log_retention.py --dry-run  # show what would be done
    env/bin/python log_retention.py --quiet    # only print summary

Intended to be scheduled via cron, e.g.:
    0 3 * * *  cd /home/ubuntu/chatbot_testing && env/bin/python log_retention.py --quiet >> logs/retention.log 2>&1
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
DIAG_DIR = BASE_DIR / "diagnostics"
SHOT_DIR = BASE_DIR / "screenshots"

KEEP_LOG_DAYS = int(os.getenv("KEEP_LOG_DAYS", "14"))
GZIP_LOG_DAYS = int(os.getenv("GZIP_LOG_DAYS", "3"))
DELETE_LOG_DAYS = int(os.getenv("DELETE_LOG_DAYS", "30"))
KEEP_DIAG_DAYS = int(os.getenv("KEEP_DIAG_DAYS", "3"))
KEEP_SCREENSHOT_DAYS = int(os.getenv("KEEP_SCREENSHOT_DAYS", "3"))
# Analytics JSONL files are consumed by downstream partners; keep them
# longer than normal logs so late backfills are possible. Gzip after
# N days to keep disk usage bounded.
KEEP_ANALYTICS_DAYS = int(os.getenv("KEEP_ANALYTICS_DAYS", "30"))
GZIP_ANALYTICS_DAYS = int(os.getenv("GZIP_ANALYTICS_DAYS", "7"))

# Matches filenames like prepare_test_runs_20260415.log or
# prepare_test_runs_20260415.log.1 (rotated peers).
LOG_NAME_RE = re.compile(r"^(?P<base>.+?)_(?P<date>\d{8})\.log(?:\.(?P<rot>\d+))?(?P<gz>\.gz)?$")

# Matches filenames like 20260415_152155_REPORT.txt,
# 20260415_152155_console_logs.json, 20260415_152155_10_turn1_TIMEOUT_dom.html,
# 20260415_152426_06_first_message_received_dom.html
DIAG_DATE_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_")

# Matches screenshot filenames like debug_timeout_turn_1_20260130_151939.png
SHOT_DATE_RE = re.compile(r"(?P<date>\d{8})_\d{6}\.png$")


def parse_date(date_str: str) -> Optional[datetime]:  # type: ignore
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None


# ---- fallback for Python <3.10 type annotation ----
try:
    from typing import Optional
except ImportError:  # pragma: no cover
    Optional = lambda x: x  # type: ignore


def days_ago(n: int) -> datetime:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=n)


def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024:
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} PB"


class Action:
    def __init__(self, kind: str, path: Path, reason: str, size: int) -> None:
        self.kind = kind  # "delete", "gzip"
        self.path = path
        self.reason = reason
        self.size = size

    def __repr__(self) -> str:
        return f"<{self.kind} {self.path.name} ({human_bytes(self.size)}) — {self.reason}>"


def plan_log_actions() -> List[Action]:
    """Decide gzip/delete actions for LOG_DIR."""
    actions: List[Action] = []
    if not LOG_DIR.exists():
        return actions

    gzip_cutoff = days_ago(GZIP_LOG_DAYS)
    delete_cutoff = days_ago(DELETE_LOG_DAYS)

    for p in LOG_DIR.iterdir():
        if not p.is_file():
            continue
        m = LOG_NAME_RE.match(p.name)
        if not m:
            continue

        date = parse_date(m.group("date"))
        if date is None:
            continue

        try:
            size = p.stat().st_size
        except Exception:
            continue

        is_gzipped = bool(m.group("gz"))

        if is_gzipped and date < delete_cutoff:
            actions.append(
                Action(
                    "delete",
                    p,
                    f"gzipped and older than {DELETE_LOG_DAYS} days",
                    size,
                )
            )
            continue

        if not is_gzipped and date < gzip_cutoff:
            actions.append(
                Action(
                    "gzip",
                    p,
                    f"older than {GZIP_LOG_DAYS} days",
                    size,
                )
            )
            continue

    return actions


def plan_diag_actions() -> List[Action]:
    """Delete entire diagnostics files older than KEEP_DIAG_DAYS."""
    actions: List[Action] = []
    if not DIAG_DIR.exists():
        return actions

    cutoff = days_ago(KEEP_DIAG_DAYS)
    for p in DIAG_DIR.iterdir():
        if not p.is_file():
            continue
        m = DIAG_DATE_RE.match(p.name)
        if not m:
            continue
        date = parse_date(m.group("date"))
        if date is None:
            continue
        if date < cutoff:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            actions.append(
                Action(
                    "delete",
                    p,
                    f"diagnostic older than {KEEP_DIAG_DAYS} days",
                    size,
                )
            )
    return actions


def plan_analytics_actions() -> List[Action]:
    """Gzip + delete old analytics JSONL files under logs/analytics/."""
    actions: List[Action] = []
    analytics_dir = LOG_DIR / "analytics"
    if not analytics_dir.exists():
        return actions

    gzip_cutoff = days_ago(GZIP_ANALYTICS_DAYS)
    delete_cutoff = days_ago(KEEP_ANALYTICS_DAYS)

    # Match events_YYYYMMDD.txt and events_YYYYMMDD.txt.gz (current format)
    # plus the legacy .jsonl naming in case old files are around.
    events_re = re.compile(r"^events_(?P<date>\d{8})\.(?:txt|jsonl)(?P<gz>\.gz)?$")

    for p in analytics_dir.iterdir():
        if not p.is_file():
            continue
        m = events_re.match(p.name)
        if not m:
            continue
        date = parse_date(m.group("date"))
        if date is None:
            continue
        try:
            size = p.stat().st_size
        except Exception:
            continue
        is_gzipped = bool(m.group("gz"))

        if is_gzipped and date < delete_cutoff:
            actions.append(
                Action("delete", p, f"analytics gzipped and older than {KEEP_ANALYTICS_DAYS} days", size)
            )
        elif not is_gzipped and date < gzip_cutoff:
            actions.append(
                Action("gzip", p, f"analytics older than {GZIP_ANALYTICS_DAYS} days", size)
            )
    return actions


def plan_screenshot_actions() -> List[Action]:
    actions: List[Action] = []
    if not SHOT_DIR.exists():
        return actions
    cutoff = days_ago(KEEP_SCREENSHOT_DAYS)
    for p in SHOT_DIR.iterdir():
        if not p.is_file():
            continue
        m = SHOT_DATE_RE.search(p.name)
        date = parse_date(m.group("date")) if m else None
        if date is None:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except Exception:
                continue
            date = mtime
        if date < cutoff:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            actions.append(
                Action(
                    "delete",
                    p,
                    f"screenshot older than {KEEP_SCREENSHOT_DAYS} days",
                    size,
                )
            )
    return actions


def execute_action(action: Action, dry_run: bool, quiet: bool) -> Tuple[int, int]:
    """Return (bytes_freed, bytes_compressed_away)."""
    if action.kind == "delete":
        if dry_run:
            if not quiet:
                print(f"[DRY] DELETE {action.path.name} ({human_bytes(action.size)}) — {action.reason}")
            return action.size, 0
        try:
            action.path.unlink()
            if not quiet:
                print(f"DELETE {action.path.name} ({human_bytes(action.size)}) — {action.reason}")
            return action.size, 0
        except Exception as e:
            print(f"ERROR deleting {action.path.name}: {e}", file=sys.stderr)
            return 0, 0

    if action.kind == "gzip":
        gz_path = Path(str(action.path) + ".gz")
        if dry_run:
            if not quiet:
                print(f"[DRY] GZIP {action.path.name} ({human_bytes(action.size)}) — {action.reason}")
            return 0, int(action.size * 0.85)  # estimated savings
        try:
            with action.path.open("rb") as fin, gzip.open(gz_path, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            compressed_size = gz_path.stat().st_size
            savings = action.size - compressed_size
            action.path.unlink()
            if not quiet:
                print(
                    f"GZIP   {action.path.name} "
                    f"({human_bytes(action.size)} → {human_bytes(compressed_size)}, "
                    f"saved {human_bytes(savings)}) — {action.reason}"
                )
            return 0, savings
        except Exception as e:
            print(f"ERROR gzipping {action.path.name}: {e}", file=sys.stderr)
            # If we failed partway, clean up any partial .gz
            try:
                gz_path.unlink()
            except Exception:
                pass
            return 0, 0

    return 0, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing")
    parser.add_argument("--quiet", action="store_true", help="Only print summary")
    args = parser.parse_args()

    start = time.time()
    if not args.quiet:
        print(f"▶ Log retention sweep starting @ {datetime.now().isoformat(timespec='seconds')}")
        print(f"  LOG_DIR={LOG_DIR}")
        print(f"  DIAG_DIR={DIAG_DIR}")
        print(f"  SHOT_DIR={SHOT_DIR}")
        print(
            f"  policy: gzip>{GZIP_LOG_DAYS}d, delete_gz>{DELETE_LOG_DAYS}d, "
            f"diag>{KEEP_DIAG_DAYS}d, shots>{KEEP_SCREENSHOT_DAYS}d"
        )
        print()

    actions = (
        plan_log_actions()
        + plan_diag_actions()
        + plan_screenshot_actions()
        + plan_analytics_actions()
    )
    if not actions:
        if not args.quiet:
            print("(nothing to do)")
        else:
            print(f"retention: nothing to do at {datetime.now().isoformat(timespec='seconds')}")
        return 0

    total_deleted = 0
    total_saved = 0
    counts = {"delete": 0, "gzip": 0}
    for a in actions:
        freed, saved = execute_action(a, args.dry_run, args.quiet)
        total_deleted += freed
        total_saved += saved
        counts[a.kind] += 1

    elapsed = time.time() - start
    summary = (
        f"retention: {counts['delete']} deleted, {counts['gzip']} gzipped, "
        f"freed {human_bytes(total_deleted)}, "
        f"saved {human_bytes(total_saved)} via compression, "
        f"in {elapsed:.1f}s"
    )
    if args.dry_run:
        summary = "[DRY RUN] " + summary
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
