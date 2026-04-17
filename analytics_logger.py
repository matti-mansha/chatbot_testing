"""
Structured analytics event log.

Writes one JSON object per line to logs/analytics/events_YYYYMMDD.jsonl.
Each line represents one event in the test pipeline. All events share a
common header (event_type, ts, execution_id, run_number, test_case,
persona) so downstream analysts can join event types on execution_id.

Downstream consumers can load the day's events with one line:

    import pandas as pd
    df = pd.read_json('logs/analytics/events_20260417.jsonl', lines=True)

Then filter/group by event_type:

    ev = df[df.event_type == 'evaluation_completed']
    ev[['run_number', 'test_case', 'overall_score', 'overall_result']]

Schema
------
See docs/ANALYTICS_SCHEMA.md for the full per-event field list.

Design notes
------------
* All writes are atomic per line because we open-append-close inside a
  single fcntl.LOCK_EX region. Safe across the bridge subprocess, the
  parent run_test_executions, and the run_test_evaluations service
  writing to the same file simultaneously.
* emit_event() NEVER raises. Analytics failures must not block the
  test pipeline. All exceptions are swallowed and printed to stderr.
* Events include a random `event_id` so idempotent downstream loads
  can deduplicate (in case the analytics consumer replays the file).
* Timestamps are ISO 8601 UTC with microsecond precision.
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

_BASE_DIR = pathlib.Path(__file__).parent
_ANALYTICS_DIR = _BASE_DIR / "logs" / "analytics"


def _today_path() -> pathlib.Path:
    _ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    return _ANALYTICS_DIR / f"events_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"


def _json_default(v: Any) -> Any:
    """Best-effort serializer for non-stdlib types so dump never raises."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (set, tuple)):
        return list(v)
    if hasattr(v, "__dict__"):
        try:
            return v.__dict__
        except Exception:
            pass
    return str(v)


def emit_event(event_type: str, **fields: Any) -> None:
    """
    Append one structured event to today's analytics JSONL file.

    Adds ``event_type``, ``event_id`` (UUID4), and ``ts`` (ISO 8601 UTC)
    automatically. All other fields are caller-supplied.

    Never raises. If the disk is full or the file is unwritable, the
    error is printed to stderr and the function returns silently — the
    pipeline keeps running.
    """
    try:
        event = {
            "event_type": event_type,
            "event_id": uuid.uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        event.update(fields)
        line = json.dumps(event, ensure_ascii=False, default=_json_default)

        path = _today_path()
        with open(path, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Non-POSIX platforms — fall back to best-effort append.
                pass
            try:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
    except Exception as e:
        # Never let analytics logging block the pipeline.
        try:
            print(f"[analytics_logger] failed to emit {event_type}: {e}", file=sys.stderr)
        except Exception:
            pass


def current_log_path() -> pathlib.Path:
    """Return the absolute path of today's analytics JSONL file.

    Useful for the dashboard's 'Download analytics' feature or for the
    log-retention sweeper.
    """
    return _today_path()
