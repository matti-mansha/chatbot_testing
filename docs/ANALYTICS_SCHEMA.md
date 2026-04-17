# Analytics event log — schema reference

The MILA chatbot testing pipeline emits structured events to:

```
logs/analytics/events_YYYYMMDD.jsonl
```

One JSON object per line. One file per UTC day. Safe to tail, rsync, or
bulk-load. Used by the analytics partner to compute bespoke metrics on
top of what the operational dashboard already shows.

## Quick start

### Python / pandas

```python
import pandas as pd

df = pd.read_json('events_20260417.jsonl', lines=True)

# All 14 aggregate KPIs can be computed from evaluation_completed rows
evals = df[df.event_type == 'evaluation_completed']
evals[['run_number', 'test_case', 'overall_score', 'overall_result']]
```

### jq / shell

```bash
# All evaluations today, with score + result
jq -c 'select(.event_type=="evaluation_completed") | {run_number, test_case, overall_score, overall_result}' \
    events_$(date +%Y%m%d).jsonl

# All MILA metadata emitted so far (per-turn, shipped once MILA emits [[...]])
jq -c 'select(.event_type=="turn_recorded" and (.mila_metadata|length>0)) | {run_number, test_case, turn_number, mila_metadata}' \
    events_*.jsonl
```

### duckdb

```sql
CREATE VIEW evals AS
  SELECT * FROM read_json_auto('logs/analytics/events_*.jsonl', format='newline_delimited')
  WHERE event_type = 'evaluation_completed';

SELECT test_case, AVG(overall_score), COUNT(*)
FROM evals
GROUP BY test_case
ORDER BY 2 ASC;
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
| `terminal_status` | string | The `Test Execution Status` value written to Notion: `"Test Executed"` on success, `"Test Execustion Started"` (sic) or another configured value on failure |

### `evaluation_completed`

Emitted by `run_test_evaluations.py` after OpenAI scores the
conversation. **This is the row analytics partners want most** — it
contains the full 9-KPI rubric + overall verdict in a single line.
One per evaluation (some executions may not reach evaluation if they
failed).

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
```json
{"score": 72, "comment": "..."}
```

## Event lifecycle

For a single test case execution the stream looks like:

```
{event_type: "execution_started",  turn:-, ts:T0, execution_id:E}
{event_type: "turn_recorded",      turn:1, ts:T0+30s, execution_id:E, mila_metadata:{}}
{event_type: "turn_recorded",      turn:2, ts:T0+60s, execution_id:E, mila_metadata:{"zielerreichung":7}}
{event_type: "turn_recorded",      turn:3, ts:T0+90s, execution_id:E, mila_metadata:{"zielerreichung":8}}
... (up to MAX_TURNS or early-exit)
{event_type: "execution_completed", turn:-, ts:T0+5m, execution_id:E, outcome:"success", num_turns:5}
{event_type: "evaluation_completed",turn:-, ts:T0+6m, execution_id:E, overall_score:72, kpis:{...}}
```

Failed executions skip the `evaluation_completed` event. They still emit
`execution_started` and `execution_completed` (with `outcome:"failed"`
and usually partial turn records up to the failure point).

## Retention

`log_retention.py` runs daily and deletes `events_*.jsonl` older than
`KEEP_ANALYTICS_DAYS` (default **30 days** — longer than normal logs
since analytics consumers may backfill). Copy or rsync to a
long-term-retention bucket if you need beyond 30 days.

## Stability guarantees

- Adding NEW fields to an event type is a non-breaking change.
- Renaming or removing existing fields is breaking; ask before doing it.
- `event_id` is suitable as a unique key for idempotent ingestion.
- `execution_id` is stable per execution; join on it across event types.
- `ts` is monotonically non-decreasing *within* a single process but
  may be slightly out-of-order across bridge subprocess vs. parent
  process events (µs-scale). Consumers should sort on `ts` if ordering
  is material.
