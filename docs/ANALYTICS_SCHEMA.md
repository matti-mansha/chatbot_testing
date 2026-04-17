# Analytics event log — schema reference

The MILA chatbot testing pipeline emits structured events to:

```
logs/analytics/events_YYYYMMDD.txt
```

Each event is a pretty-printed JSON object wrapped in **double square
brackets** (`[[ ... ]]`), per the project-agreed chat-metadata format
(from the "Website chat - add chat metadata" issue). Events are
separated by a blank line.

Example event:

```
[[
{
  "event_type": "evaluation_completed",
  "event_id": "f53a2daa24d54470805020242fdbdd29",
  "ts": "2026-04-17T12:35:54.719629+00:00",
  "execution_id": "343af83c-f3e4-81eb-941d-dd4bdca938cd",
  "run_number": "TR1.TC60.5",
  "test_case": "Partial info: name known, email missing",
  "persona": "Cooperative user",
  "overall_score": 72,
  "overall_result": "PARTIAL",
  "overall_comment": "...",
  "summary_of_goal": "...",
  "kpis": {
    "task_completeness": { "score": 75, "comment": "..." },
    "user_comfort": { "score": 80, "comment": "..." },
    ...
  },
  "evaluation_page_url": "https://www.notion.so/...",
  "evaluator_duration_sec": 29.3,
  "evaluator_error": null
}
]]
```

One file per UTC day. Safe to tail, rsync, or bulk-load.

## Downloading

The files are served over HTTPS from the dashboard host, gated by the
same Basic Auth credentials as the dashboard itself:

```
https://<host>/analytics/                            # directory listing
https://<host>/analytics/events_YYYYMMDD.txt         # today's file
```

## Parsing

Because every event is a `[[ ... ]]`-wrapped JSON object, a one-line
regex extracts all events from a day's file:

### Python (pandas)

```python
import re, json, pandas as pd

with open('events_20260417.txt') as f:
    raw = f.read()

blocks = re.findall(r'\[\[\s*(\{.*?\})\s*\]\]', raw, re.DOTALL)
events = [json.loads(b) for b in blocks]
df = pd.DataFrame(events)

# Only evaluation results (one row per evaluated execution)
evals = df[df.event_type == 'evaluation_completed']
print(evals[['run_number','test_case','overall_score','overall_result']])
```

### Shell (awk + jq)

```bash
# Extract every JSON block from a day's file and pipe into jq
awk '/^\[\[/ {inblock=1; next} /^\]\]/ {inblock=0; print ""; next} inblock' \
    events_20260417.txt \
| jq -s 'map(select(.event_type=="evaluation_completed"))'
```

### Multi-day bulk load

```python
import glob, re, json, pandas as pd
all_events = []
for path in glob.glob('events_*.txt'):
    raw = open(path).read()
    for b in re.findall(r'\[\[\s*(\{.*?\})\s*\]\]', raw, re.DOTALL):
        all_events.append(json.loads(b))
df = pd.DataFrame(all_events)
```

## Common fields (every event)

| Field | Type | Description |
|---|---|---|
| `event_type` | string | One of: `execution_started`, `turn_recorded`, `execution_completed`, `evaluation_completed` |
| `event_id` | string (hex) | UUID4 — unique per event. Use for idempotent re-ingestion |
| `ts` | string (ISO 8601 UTC) | Emit time, microsecond precision |
| `execution_id` | string | Notion page ID of the Test Case Execution row. **Primary join key across event types.** |
| `run_number` | string | e.g. `TR1.TC60.5`. Empty for events before the execution is claimed |
| `test_case` | string | Human-readable test case name |
| `persona` | string | Persona label from the test case |

## Per-event-type fields

### `execution_started`

Emitted by `run_test_executions.py` the moment an execution row is
claimed from the queue. Marks the beginning of a single Playwright
bridge subprocess run.

| Field | Type | Description |
|---|---|---|
| `test_case_prompt_len` | int | Character length of the OpenAI system prompt |
| `test_case_details_len` | int | Character length of the test case details |

### `turn_recorded`

Emitted by `playwright_bridge_bot_headless.py` after each successful
conversation turn (tester sends message → MILA replies). One event
per turn, per successful attempt. **Not emitted for attempts that end
in restart / failure.**

| Field | Type | Description |
|---|---|---|
| `turn_number` | int | 1-indexed turn counter within the attempt |
| `max_turns` | int | Configured MAX_TURNS (default 15) |
| `user_message` | string | What the tester bot sent to MILA (may be long) |
| `user_message_len` | int | Character length of user_message |
| `mila_reply` | string | MILA's reply WITH `[[...]]` metadata already stripped |
| `mila_reply_len` | int | Character length of mila_reply |
| `mila_metadata` | object | Parsed JSON from the `[[{...}]]` block in MILA's raw reply. Empty `{}` if no metadata block present. Per the "Website chat - add chat metadata" spec, this is where fields like `zielerreichung` will appear once MILA ships the feature. |
| `tester_completeness_score` | int \| null | 1-100, self-assessment from the tester bot |
| `tester_should_continue` | bool | Does the tester bot think the conversation should continue |

### `execution_completed`

Emitted by `run_test_executions.py` after the bridge subprocess exits
(success or failure). One per execution.

| Field | Type | Description |
|---|---|---|
| `outcome` | string | `"success"` or `"failed"` |
| `num_turns` | int \| null | Turns that ran before exit |
| `duration_sec` | float | Wall-clock seconds from bridge spawn to exit |
| `per_turn_scores` | array of int | Per-turn completeness scores from the tester bot (ordered by turn) |
| `conversation_page_id` | string \| null | Notion page ID of the formatted conversation page |
| `conversation_page_url` | string \| null | Browser URL of the conversation page |
| `terminal_status` | string | The `Test Execution Status` value written to Notion |

### `evaluation_completed`

Emitted by `run_test_evaluations.py` after OpenAI scores the
conversation. **This is the row analytics partners want most** — it
contains the full 9-KPI rubric + overall verdict in a single event.

| Field | Type | Description |
|---|---|---|
| `overall_score` | int | 1-100 |
| `overall_result` | string | `"PASS"`, `"PARTIAL"`, or `"FAIL"` |
| `overall_comment` | string | One paragraph, strengths + improvements |
| `summary_of_goal` | string | 1-3 sentences describing user goals |
| `kpis` | object | Dict of 9 dimensions → `{score: int, comment: str}` |
| `evaluation_page_id` | string \| null | Notion page ID of the child evaluation page |
| `evaluation_page_url` | string \| null | Browser URL |
| `evaluator_duration_sec` | float | Wall-clock seconds the OpenAI call took |
| `evaluator_error` | string \| null | Populated when OpenAI failed or returned unparseable JSON |

The `kpis` field contains these 9 keys (from the `EvaluationOutput`
Pydantic model):

- `task_completeness`
- `user_comfort`
- `understanding_and_relevance`
- `clarity_and_actionability`
- `edge_cases_and_constraints`
- `proactiveness_and_guidance`
- `tone_and_personalization`
- `accuracy_and_policy_compliance`
- `efficiency_and_flow`

Each value is an object:
```
{ "score": 72, "comment": "..." }
```

## Event lifecycle

For a single test case execution the stream looks like (separator
blank lines omitted):

```
[[ { event_type: "execution_started",   turn:-, ts:T0,      execution_id:E } ]]
[[ { event_type: "turn_recorded",       turn:1, ts:T0+30s,  execution_id:E, mila_metadata:{} } ]]
[[ { event_type: "turn_recorded",       turn:2, ts:T0+60s,  execution_id:E, mila_metadata:{"zielerreichung":7} } ]]
... (up to MAX_TURNS or early-exit)
[[ { event_type: "execution_completed", turn:-, ts:T0+5m,   execution_id:E, outcome:"success", num_turns:5 } ]]
[[ { event_type: "evaluation_completed",turn:-, ts:T0+6m,   execution_id:E, overall_score:72, kpis:{...} } ]]
```

Failed executions skip the `evaluation_completed` event. They still emit
`execution_started` and `execution_completed` (with `outcome:"failed"`
and usually partial turn records up to the failure point).

## Retention

`log_retention.py` runs daily and:
- Gzips `events_*.txt` older than `GZIP_ANALYTICS_DAYS` (default 7).
- Deletes `events_*.txt.gz` older than `KEEP_ANALYTICS_DAYS` (default 30).

Copy or rsync to a long-term-retention bucket if you need beyond 30 days.

## Stability guarantees

- Adding NEW fields to an event type is a non-breaking change.
- Renaming or removing existing fields is breaking; ask before doing it.
- `event_id` is suitable as a unique key for idempotent ingestion.
- `execution_id` is stable per execution; join on it across event types.
- The `[[ ... ]]` wrapper and blank-line separator are part of the format
  contract — parsers should rely on them.
- `ts` is monotonically non-decreasing *within* a single process but
  may be slightly out-of-order across bridge subprocess vs. parent
  process events (µs-scale). Consumers should sort on `ts` if ordering
  is material.
