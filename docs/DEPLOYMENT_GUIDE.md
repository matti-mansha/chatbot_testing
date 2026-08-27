# MILA Chatbot Testing Framework — Deployment Guide

**Every command below is copy-paste ready.** Run them in order on a fresh
Ubuntu 24.04 host.

- **Repository:** https://github.com/matti-mansha/chatbot_testing (public — no token needed)
- **Notion workspace:** AuPairWorld → *Chatbot Handover*
- **Reference host:** `i-06ea8478ce16a7714` · `52.29.135.98` · `eu-central-1` · t3.small · 30 GB

---

## 0. Requirements

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 LTS |
| RAM | 2 GB minimum (headless Chromium needs ~1 GB headroom) |
| Disk | 25 GB minimum |
| Outbound | `api.openai.com`, `api.notion.com`, `stage.aupairworld.com`, `github.com` |
| Inbound | SSH (22) only — **no service port needs to be exposed** |

You will also need: an **OpenAI API key**, a **Notion integration token** for the
AuPairWorld workspace, and the **MILA QA account** credentials.

---

## 1. Connect

```bash
ssh -i /path/to/your-key.pem ubuntu@52.29.135.98
```

---

## 2. System packages

```bash
sudo apt-get update -y && sudo apt-get upgrade -y && sudo apt-get install -y python3-venv python3-pip git curl unzip
```

---

## 3. Clone the application

```bash
git clone https://github.com/matti-mansha/chatbot_testing.git ~/chatbot_testing && cd ~/chatbot_testing
```

---

## 4. Virtual environment and Python dependencies

```bash
cd ~/chatbot_testing && python3 -m venv env && ./env/bin/pip install --upgrade pip && ./env/bin/pip install -r requirements.txt
```

---

## 5. Headless Chromium

Two commands. The first needs `sudo` (it installs ~50 system libraries); the
second must run **without** sudo so the browser lands in the app user's cache.

```bash
cd ~/chatbot_testing && sudo ./env/bin/playwright install-deps chromium
```

```bash
cd ~/chatbot_testing && ./env/bin/playwright install chromium
```

---

## 6. Configuration

Create `.env`. Replace the three `REPLACE_ME` values — everything else is correct
for the AuPairWorld workspace as written.

```bash
cd ~/chatbot_testing && cat > .env <<'EOF'
# --- OpenAI (drives both the tester bot and the evaluator) ---
OPENAI_API_KEY=REPLACE_ME
OPENAI_MODEL=gpt-5-nano
OPENAI_TIMEOUT=100

# --- Notion (AuPairWorld workspace: Chatbot Handover) ---
NOTION_API_KEY=REPLACE_ME
TEST_CASES_DB_ID=3c229027a74281e4bfb8ebc8fc0574a3
TEST_RUNS_DB_ID=3c229027a742818a9c36de754542c624
TEST_CASE_EXECUTIONS_DB_ID=3c229027a742815da28cd93d1edde0d1
CONVERSATIONS_PARENT_PAGE_ID=3c229027a74281828057f94c893239e5
EVALUATION_PAGES_DB_ID=3c229027a742813eb47add26917b8dfa
TEST_CASE_PROMPT_PARENT_PAGE_ID=3c229027a74281219c46c255541ee25d
EVALUATION_PROMPT_PARENT_PAGE_ID=3c229027a742810e8c47f4a9f20f0324

# Notion property names (case-insensitive match)
EVALUATION_TEXT_PROPERTY=Evaluation JSON
EVALUATION_SCORE_PROPERTY=Evaluation Score
EVALUATION_PAGE_LINK_PROPERTY=

# Status values - must match Notion EXACTLY.
# "Test Execustion Started" is a real typo in the Notion schema. Do not correct it.
EXECUTION_SUCCESS_STATUS=Test Executed
EXECUTION_FAILED_STATUS=Test Execustion Started

# --- MILA (chatbot under test) ---
MILA_URL=https://stage.aupairworld.com/en/user/login
MILA_LOGIN_USER=REPLACE_ME
MILA_LOGIN_PASS=REPLACE_ME
MILA_HTTP_USER=apw
MILA_HTTP_PASS=apw

# --- Service wiring ---
TESTER_API_URL=http://localhost:8501
TESTER_API_TIMEOUT=120
BRIDGE_SCRIPT=playwright_bridge_bot_headless.py

# --- Pipeline behaviour ---
MAX_TURNS=15
EXECUTION_CHECK_INTERVAL=10
EVALUATION_CHECK_INTERVAL=10
SESSION_TIMEOUT_MINUTES=30
STALE_EVAL_CLAIM_MINUTES=5
MAX_RETRIES=3
RETRY_DELAY=5
MAX_TEST_RESTARTS=5
RESTART_DELAY=10
MAX_MILA_REJECTION_RETRIES=1

# --- Logging ---
LOG_LEVEL=INFO
LOG_DIR=logs
DETAILED_TIMING=false
ENABLE_SCREENSHOTS=false

# --- Log retention (cron: 0 3 * * *) ---
KEEP_LOG_DAYS=14
GZIP_LOG_DAYS=3
DELETE_LOG_DAYS=30
KEEP_DIAG_DAYS=3
KEEP_SCREENSHOT_DAYS=3
KEEP_ANALYTICS_DAYS=30
GZIP_ANALYTICS_DAYS=7
EOF
chmod 600 .env && echo ".env created"
```

Then edit the three placeholders:

```bash
cd ~/chatbot_testing && nano .env
```

> **The Notion integration must be shared with every page above.** In Notion open
> *Chatbot Handover* → `•••` → **Connections** → add your integration. An ID that is
> correct but unshared returns 404 and the pipeline fails **silently**.

---

## 7. Validate before starting

```bash
cd ~/chatbot_testing && ./env/bin/python preflight.py
```

This resolves all 7 Notion objects, checks the property and status names against
the live schema, tests the OpenAI key, launches Chromium, and reaches MILA.

**Do not continue until this is green.** It exists to catch the misconfiguration
class that previously ran undetected for three months.

---

## 8. Start the pipeline

```bash
cd ~/chatbot_testing && ./start_services.sh && ./testbot status
```

Expected — four services `alive`:

```
SERVICE      PID      UPTIME     STATE
● tester_bot 1234     00:06      alive
● prepare    1235     00:06      alive
● execute    1236     00:03      alive
● evaluate   1237     00:03      alive
```

---

## 9. Nightly log cleanup

```bash
(crontab -l 2>/dev/null; echo "0 3 * * * cd ~/chatbot_testing && ./env/bin/python log_retention.py --quiet >> logs/retention.log 2>&1") | crontab -
```

Without this, logs grow without bound.

---

# Operator Guide — running a test from Notion

No terminal access is required for day-to-day use.

## Trigger a test run

1. Open **Chatbot Handover → Databases → Test Runs** in Notion
2. Create a new row
3. Set **Test Run Number** (e.g. `TR5`)
4. Set **Status** to **`Prepare Test Run`**

The pipeline picks it up within ~10 seconds and does the rest.

## What happens next

| Stage | What you see in Notion |
|---|---|
| 1. Prepare | Status moves to `Start Test Run`; one row per active test case appears in **Test Case Executions** |
| 2. Execute | Each execution moves to `Test Execustion Started`, then `Test Executed`. A **Conversation** page is created with the full transcript |
| 3. Evaluate | Status moves to `Evaluation started` → `Evaluation completed`. **Evaluation Score** and **Evaluation JSON** are filled in |

Roughly **3–5 minutes per test case**. Execution is **strictly serial** — a single
shared MILA account provides no conversation isolation — so a 50-case suite takes
several hours.

## Choose which test cases run

In **Test cases DB**, set each row's **Status**:

- **`Active`** → included in every future run
- **`Inactive`** → skipped, but kept for history

## Change the prompts (no deployment needed)

Prompts are versioned as child pages under **Test Prompts** and **Evaluation
Prompt**. Add a new child page and the pipeline automatically uses the newest one.
No code change, no restart.

## Read the results

Each execution links to:

- a **Conversation** page — the full transcript, turn by turn
- an **Evaluation** page — 9 KPI scores, comments, and an overall PASS / PARTIAL / FAIL

## Export the analytics

```bash
ls -la ~/chatbot_testing/logs/analytics/
```

`events_YYYYMMDD.jsonl` is the **durable system of record** and what the analytics
vendor consumes. **Always export it before clearing anything in Notion.**

---

# Day-to-day commands

```bash
cd ~/chatbot_testing && ./testbot status
```

```bash
cd ~/chatbot_testing && ./testbot errors
```

```bash
cd ~/chatbot_testing && tail -f logs/execute.log
```

```bash
cd ~/chatbot_testing && ./start_services.sh
```

```bash
cd ~/chatbot_testing && ./stop_services.sh
```

> Services are started by a shell script and **do not survive a reboot**. After any
> restart of the host, run `./start_services.sh` again.

---

# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Found 0 test runs to prepare` repeating | Normal when no run is queued | Set a Test Runs row to `Prepare Test Run` |
| `Found 0 pending executions` while rows exist | Execution rows have a blank **Test Execution Status** | Fixed in the current release. On older code, set the cell to `Not started` manually |
| `404 object_not_found` | A Notion page is not shared with the integration | Notion → `•••` → Connections → add it. Then run `preflight.py` |
| `LOGIN FAILED … Unrecognized username or password` | Wrong MILA credentials | Correct `MILA_LOGIN_PASS` |
| `LOGIN FAILED … failed login attempts` | Drupal flood control after repeated failures | Wait ~1 hour, or ask AuPairWorld to clear the flood table |
| `intercepts pointer events` | A Drupal notice banner covering the chat button | Handled automatically in the current release |
| Chromium will not start | `install-deps` was never run | Re-run step 5 |
| OpenAI quota errors | Credit exhausted | Top up; set a spend limit |

Run the bridge self-test at any time — it makes no network calls and never touches
the live MILA site:

```bash
cd ~/chatbot_testing && ./env/bin/python tools/test_bridge_overlay_fixes.py
```

---

# Security notes

- `.env` holds live credentials in plaintext. Keep it `chmod 600`; never commit it.
- The tester API binds `0.0.0.0:8501` **without authentication**. Do not expose port
  8501 in the security group or firewall — anything that can reach it can spend your
  OpenAI budget.
- Use a **dedicated QA account** for MILA, never a staff login.
