"""
Structured analytics event log.

Writes events to logs/analytics/events_YYYYMMDD.txt using the
project-agreed chat-metadata format (per the "Website chat - add chat
metadata" issue): each event is a pretty-printed JSON object wrapped
in double square brackets, e.g.

    [[
    {
      "event_type": "evaluation_completed",
      "event_id": "f53a2daa...",
      "ts": "2026-04-17T12:35:54.719629+00:00",
      "execution_id": "abc-123",
      "run_number": "TR1.TC60.5",
      "test_case": "Partial info: name known, email missing",
      "persona": "Cooperative user",
      "overall_score": 72,
      "overall_result": "PARTIAL",
      "kpis": { ... }
    }
    ]]

Events are appended in order, separated by blank lines. One file per
UTC day. Safe to tail, rsync, or bulk-load.

Parsing a day's file
--------------------

Because the format is "JSON object wrapped in `[[ ... ]]`", a consumer
can split on `]]` + newline, or use a tiny regex:

    import re, json
    with open('events_20260417.txt') as f:
        raw = f.read()
    # Find every [[...JSON...]] block
    blocks = re.findall(r'\\[\\[\\s*(\\{.*?\\})\\s*\\]\\]', raw, re.DOTALL)
    events = [json.loads(b) for b in blocks]

Or simpler with pandas:

    events = [json.loads(b) for b in re.findall(
        r'\\[\\[\\s*(\\{.*?\\})\\s*\\]\\]', open('events_*.txt').read(), re.DOTALL)]
    df = pd.DataFrame(events)

Schema
------
See docs/ANALYTICS_SCHEMA.md for the full per-event field list.

Design notes
------------
* All writes are atomic (one full `[[ ... ]]` block) inside a single
  fcntl.LOCK_EX region. Safe across the bridge subprocess, the parent
  run_test_executions, and the run_test_evaluations service writing
  concurrently.
* emit_event() NEVER raises. Analytics failures must not block the
  test pipeline. All exceptions are swallowed and printed to stderr.
* Events include a random `event_id` so idempotent downstream loads
  can deduplicate.
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
    return _ANALYTICS_DIR / f"events_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"


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
    Append one structured event to today's analytics file in the
    project-agreed chat-metadata format:

        [[
        {
          "event_type": "...",
          ...
        }
        ]]

    Followed by a blank line so consecutive events are visually
    separated in the raw file.

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

        # Pretty-print with 2-space indent so the file is human-readable
        # at a glance, matching the format shown in the GitHub issue.
        json_body = json.dumps(
            event,
            ensure_ascii=False,
            default=_json_default,
            indent=2,
            sort_keys=False,
        )
        block = f"[[\n{json_body}\n]]\n\n"

        path = _today_path()
        with open(path, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Non-POSIX platforms — fall back to best-effort append.
                pass
            try:
                f.write(block)
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
