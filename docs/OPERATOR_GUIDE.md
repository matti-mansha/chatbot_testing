# MILA Chatbot Testing — Operator Guide

Everything you need to run, monitor, debug, and extend the pipeline.
Bookmark this one doc.

---

## 1. Quick reference — URLs and credentials

| What | Where | Credentials |
|---|---|---|
| **Dashboard** | https://3.122.134.15/ | `admin` / `J5GyV9NiWcR1DXg4yBev` |
| **Analytics downloads** | https://3.122.134.15/analytics/ | same as above |
| **Health check** | https://3.122.134.15/health | (no auth) |
| **EC2 SSH** | `ssh -i ~/downloads/Matti-aupair.pem ubuntu@ec2-3-122-134-15.eu-central-1.compute.amazonaws.com` | SSH key |
| **Repo** | https://github.com/matti-mansha/chatbot_testing | GitHub |
| **Notion test runs DB** | Your Notion workspace → "Test Runs" | your Notion login |
| **Notion evaluations parent** | configured via `EVALUATION_PAGES_DB_ID` in `.env` | |

To change the dashboard password:
```bash
ssh to server
cd chatbot_testing
sudo ./testbot nginx-install --password NEW_PASSWORD
```

---

## 2. Daily operator cheatsheet

You don't need to touch the server normally. If you ever do:

```bash
./testbot status            # 5-second pipeline snapshot
./testbot kill-bridge       # kill the currently-running bridge if stuck
./testbot restart           # full stop + start (after a code pull)
./testbot tail bridge 100   # tail any service log by friendly name
./testbot errors            # today's ERROR / Traceback / 🚨 across all logs
./testbot disk              # see disk usage under logs/, diagnostics/, screenshots/
./testbot retention         # one-shot log/diagnostics sweep
```

---

## 3. The pipeline — what happens when you trigger a test

1. **You flip a Notion Test Run** to status `Prepare Test Run`
2. **`prepare_test_runs.py`** (polls every 10 s) sees it → for each Active test case, creates a child row in the Test Case Executions DB with status `Not started`
3. **`run_test_executions.py`** (polls every 10 s) sees pending executions → claims one by marking it `Test Execustion Started` → spawns `playwright_bridge_bot_headless.py` as a subprocess
4. **Bridge** opens a headless Chrome, logs into MILA, opens the chat widget, runs up to 15 turns of tester↔Mila conversation. On each turn:
   - Tester bot (`test_bot_headless.py`, HTTP on 127.0.0.1:8501) produces the next user message via OpenAI
   - Bridge types it into Mila's UI, captures Mila's reply
   - Strips any `[[{...}]]` metadata block from Mila's reply (per the "Website chat - add chat metadata" spec)
   - Emits a `turn_recorded` analytics event
5. **Bridge exits** → parent uploads the formatted conversation page to Notion → marks execution `Test Executed` (success) or leaves it in `Test Execustion Started` (failure) → emits `execution_completed` analytics event
6. **`run_test_evaluations.py`** (polls every 10 s) picks up any execution in `Test Executed` → fetches the conversation page → calls OpenAI with the 9-KPI rubric → writes Evaluation Page in Notion → updates the Test Run's average score → emits `evaluation_completed` analytics event

Everything above runs 24/7 as long as the 5 services are alive.

---

## 4. Where things live — file map

### On the server (`/home/ubuntu/chatbot_testing/`)

```
chatbot_testing/
├── .env                              # all secrets: OpenAI, Notion, MILA, HTTP-auth
├── requirements.txt                  # Python deps (streamlit, openai, notion-client, playwright, …)
├── env/                              # Python virtualenv — never commit
│
├── PIPELINE SERVICES (long-running):
│   ├── test_bot_headless.py          # HTTP tester-bot server on 127.0.0.1:8501
│   ├── prepare_test_runs.py          # Notion → executions creator (polls 10 s)
│   ├── run_test_executions.py        # executions → bridge dispatcher (polls 10 s)
│   ├── run_test_evaluations.py       # Test Executed → OpenAI-scored evaluations (polls 10 s)
│   ├── dashboard_app.py              # Streamlit dashboard on 127.0.0.1:8502
│   └── playwright_bridge_bot_headless.py   # spawned by run_test_executions (one-shot per test)
│
├── SUPPORT MODULES:
│   ├── analytics_logger.py           # emits JSONL events to logs/analytics/
│   ├── calculate_kpis.py             # 14-KPI CLI report (dashboard imports its calc_* fns)
│   ├── format_conversation_page.py   # renders the Notion conversation page after each test
│   ├── diagnostic_utils.py           # screenshots, DOM captures, network captures (per-bridge)
│   └── logging_config.py             # shared logger factory
│
├── OPS SCRIPTS:
│   ├── testbot                       # bash CLI — the only tool you'll touch daily
│   ├── start_services.sh             # boots all 5 services (safe to re-run)
│   ├── stop_services.sh              # kills all 5 cleanly, including orphaned chromium
│   └── log_retention.py              # daily sweep — gzips + prunes old logs / diagnostics
│
├── nginx/
│   ├── install_nginx.sh              # one-command reverse-proxy installer (sudo)
│   ├── uninstall_nginx.sh            # undo install_nginx.sh
│   ├── mila-dashboard.conf.template  # nginx site config template
│   └── analytics_location.conf       # /analytics/ directory-serving config (spliced in)
│
├── docs/
│   ├── OPERATOR_GUIDE.md             # this file
│   └── ANALYTICS_SCHEMA.md           # schema of logs/analytics/events_*.jsonl
│
├── logs/                             # rotated logs from all services
│   ├── test_bot_headless_YYYYMMDD.log
│   ├── prepare_test_runs_YYYYMMDD.log
│   ├── test_execution_YYYYMMDD.log
│   ├── test_evaluation_YYYYMMDD.log
│   ├── playwright_bridge_YYYYMMDD.log
│   ├── format_conversation_page_YYYYMMDD.log
│   ├── dashboard.log                 # stdout/stderr from streamlit
│   └── analytics/
│       └── events_YYYYMMDD.jsonl     # structured event stream — see ANALYTICS_SCHEMA.md
│
├── diagnostics/                      # per-bridge DOM snapshots, HAR, screenshots
└── screenshots/                      # per-bridge PNG captures (ENABLE_SCREENSHOTS=true)
```

### Nginx files (`/etc/nginx/`)

```
/etc/nginx/
├── sites-available/mila-dashboard          # site config (proxy + /analytics/)
├── sites-enabled/mila-dashboard → ../…     # symlink that activates the site
├── conf.d/mila-ratelimit.conf              # rate-limit zone
└── htpasswd-mila                            # Basic Auth credentials

/etc/ssl/mila-dashboard/
├── fullchain.pem                            # self-signed cert
└── privkey.pem
```

---

## 5. Environment variables (`.env`)

Grouped by concern. Don't commit this file to git.

### OpenAI
```
OPENAI_API_KEY=sk-proj-...        # ⚠️ must have credits — exhaustion fails ALL tests
OPENAI_MODEL=gpt-5-nano            # tester bot + evaluator both use this
OPENAI_TIMEOUT=100                 # per-call timeout (seconds)
```

### Notion
```
NOTION_API_KEY=ntn_...
NOTION_PARENT_PAGE_ID=...          # parent for auto-created child dbs
TEST_CASES_DB_ID=...               # source of test cases
TEST_RUNS_DB_ID=...                # where you trigger runs
TEST_CASE_EXECUTIONS_DB_ID=...     # per-execution rows (the workhorse)
EVALUATION_PAGES_DB_ID=...         # parent page for evaluation child pages
CONVERSATIONS_PARENT_PAGE_ID=...   # parent page for per-execution conversation pages
TEST_CASE_PROMPT_PARENT_PAGE_ID=... # where the system prompt is stored
EVALUATION_PROMPT_PARENT_PAGE_ID=... # where the evaluation rubric template is stored
EVALUATION_TEXT_PROPERTY="Evaluation JSON"
EVALUATION_SCORE_PROPERTY="Evaluation Score"
EVALUATION_PAGE_LINK_PROPERTY="Evaluation Page Link"
```

### MILA staging auth
```
MILA_URL=https://stage.aupairworld.com/en/user/login
MILA_HTTP_USER=apw                 # HTTP Basic (site-wide)
MILA_HTTP_PASS=apw
MILA_LOGIN_USER=hhc@setaro.de      # the test account the bridge logs in as
MILA_LOGIN_PASS=...
```

### Pipeline tuning
```
CHECK_INTERVAL=10                  # prepare service poll interval (s)
EXECUTION_CHECK_INTERVAL=10        # execute service poll interval
EVALUATION_CHECK_INTERVAL=10       # evaluator service poll interval
MAX_TURNS=15                       # hard cap on conversation length per test
SESSION_TIMEOUT_MINUTES=10         # tester-bot session expiry
TESTER_API_TIMEOUT=120
MAX_RETRIES=3                      # per-HTTP-call retry
RETRY_DELAY=5
MAX_TEST_RESTARTS=5                # bridge-level restart-the-whole-test retries
RESTART_DELAY=10
EXECUTION_PARALLELISM=1            # serial by default (single MILA account)
MAX_MILA_REJECTION_RETRIES=1       # fail-fast on Mila backend content errors
STALE_EVAL_CLAIM_MINUTES=5         # evaluator stale-claim reaper threshold
```

### Dashboard / nginx
```
DASHBOARD_USER=admin               # (optional) consumed only by nginx-install
DASHBOARD_PASS=                    # (optional) consumed only by nginx-install
```

### Retention (used by log_retention.py)
```
KEEP_LOG_DAYS=14
GZIP_LOG_DAYS=3
DELETE_LOG_DAYS=30
KEEP_DIAG_DAYS=3
KEEP_SCREENSHOT_DAYS=3
KEEP_ANALYTICS_DAYS=30             # analytics JSONL — longer retention
GZIP_ANALYTICS_DAYS=7
```

---

## 6. Dashboard tour — `https://3.122.134.15/`

5 tabs:

1. **🚀 Live** — pulsing status pills for each service, currently-running bridge card with Turn N/15 progress, today's pass/fail donut, last 12 executions, auto-refreshes every 5 s
2. **📊 KPIs** — the full 14-KPI view organized by Quality / Efficiency / Consistency / Topic-level. Test-run selector dropdown, scorecard strip, 9-dimension bar chart, score distribution donut, worst/best-5 tables, persona fairness callout, historical trend line, per-test-case leaderboard. Also: **"📥 Today's events"** and **"📥 Yesterday"** direct-download buttons for the analytics partner
3. **📝 Logs** — service selector, regex filter, tail up to 500 lines, color-coded by level
4. **⚠️ Failures** — bar chart of failure reasons (mila_backend_rejection, tester_api_unavailable, max_restarts_exceeded, …) and drill-down of every failed execution today
5. **⚙️ Services** — detailed process table, log-file sizes, uptime per service

Auto-refresh toggle + interval slider in the sidebar (defaults: on, every 5 s).

---

## 7. The analytics event stream (for your analytics partner)

### What they get

A JSONL file per UTC day at:
```
https://3.122.134.15/analytics/events_YYYYMMDD.jsonl
```

One JSON object per line. Four event types, joined by `execution_id`:

- `execution_started` — pipeline picks up a pending execution
- `turn_recorded` — one per turn; **includes Mila's `[[{...}]]` self-reported metadata** (empty `{}` until Mila ships that feature)
- `execution_completed` — bridge subprocess exits
- `evaluation_completed` — **the money row** — full 9-KPI rubric + overall score + PASS/PARTIAL/FAIL + Notion URLs

### Full field catalog

See `docs/ANALYTICS_SCHEMA.md` — copy that doc to the analytics partner.

### Quickstart snippets

```python
import pandas as pd
df = pd.read_json('events_20260417.jsonl', lines=True)
evals = df[df.event_type == 'evaluation_completed']
print(evals[['run_number','test_case','overall_score','overall_result']])
```

```bash
curl -u admin:PASSWORD https://3.122.134.15/analytics/events_20260417.jsonl -o today.jsonl
jq -c 'select(.event_type=="evaluation_completed") | {run_number, test_case, overall_score}' today.jsonl
```

Retention: daily files kept uncompressed for 7 days, gzipped for 30 days total. Rsync to long-term storage if you need longer.

---

## 8. Deploying code changes

```bash
# on your laptop, from the repo:
git add <files>
git commit -m "..."
git push origin main

# on the server:
ssh to server
cd chatbot_testing
git pull origin main
./testbot restart             # picks up new code
./testbot status              # verify services came back
```

The bridge subprocess is re-spawned from disk on each execution, so bridge-only changes don't even need a service restart — the next execution picks them up automatically. Dashboard / service changes DO need a restart.

---

## 9. Failure modes and what they mean

| Symptom | Root cause | What to do |
|---|---|---|
| `BRIDGE_FAILURE_REASON=tester_api_unavailable` in logs | OpenAI key out of quota / billing | Top up OpenAI credits |
| `BRIDGE_FAILURE_REASON=mila_backend_rejection` | Mila returned "Error, please try again" or similar backend crash | Flaky — usually retries a different test. If ALL tests hit it, Mila staging is down |
| `MAX RESTARTS EXCEEDED` | Bridge burned 5 restart attempts without a clean completion | Check bridge log for the specific failure (Mila hang, login fail, widget broken) |
| Dashboard KPI tab shows old data | 60 s cache | Click "🔄 Refresh KPIs" or wait |
| Test stuck in "Evaluation started" > 5 min | Evaluator crashed mid-evaluation | Auto-recovered on next poll by `reap_stale_evaluation_claims` |
| Test stuck in "Test Execustion Started" | Bridge crashed or exited non-zero | Check the conversation page (prefix `[FAILED]`) and the stderr tail banner. Reset to "Not started" to retry |
| "Not Secure" browser warning | Self-signed cert | Expected. Click through once per browser. See §10 to upgrade to Let's Encrypt |

---

## 10. Known limitations / future work

- **Self-signed HTTPS** — browser shows "Not Secure". To upgrade: point a domain at `3.122.134.15`, then `sudo certbot --nginx -d your-domain.com` (already works, installer supports it)
- **Single MILA account** → `EXECUTION_PARALLELISM=1` forced. To run parallel: get multiple MILA staging accounts, implement `MILA_ACCOUNT_POOL` (design outlined in an earlier commit message)
- **No EC2 auto-start** — if the instance reboots, services are dead until someone runs `./testbot start`. Solvable with systemd units (build planned but not shipped per your preference)
- **No alerting** — a Slack/Discord webhook on failure spikes would be useful; not built yet

---

## 11. Emergency procedures

### "Everything is broken"

```bash
ssh to server
cd chatbot_testing
./testbot stop
./testbot disk            # check for full disk
./testbot errors          # check for fresh tracebacks
git status                # check for uncommitted local changes
git log --oneline -5      # check deploy state
./testbot start
./testbot status
```

### "The OpenAI key expired"

```bash
# Test whether it's actually the key's fault:
ssh to server; cd chatbot_testing
source <(grep -E '^(OPENAI_API_KEY|OPENAI_MODEL)=' .env | sed 's/^/export /')
env/bin/python -c "from openai import OpenAI; import os; \
  r = OpenAI(api_key=os.environ['OPENAI_API_KEY']).chat.completions.create( \
    model=os.environ.get('OPENAI_MODEL','gpt-4o-mini'), \
    messages=[{'role':'user','content':'hi'}]); \
  print(r.choices[0].message.content)"
# If RateLimitError → go to https://platform.openai.com/account/billing
```

### "Nginx is down / can't reach dashboard"

```bash
ssh to server
sudo systemctl status nginx       # is nginx running?
sudo nginx -t                     # is the config valid?
sudo systemctl reload nginx       # reload if config changed
./testbot nginx-status            # summary + recent access log
```

### "I want to re-run a failed execution"

1. Find the execution row in Notion's Test Case Executions DB
2. Change its `Test Execution Status` back to `Not started`
3. Within ~10 s the execute service picks it up and runs it fresh
