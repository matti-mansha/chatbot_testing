# AuPairWorld Chatbot Testing Framework — Handover Plan (Proposal)

> **Status:** Draft for internal discussion (Setaro) before presenting to AuPairWorld.
> **Author:** Matti Ur Rehman · **Date:** 2026-06-26
> **Goal:** Cleanly transfer the MILA chatbot QA framework to AuPairWorld so Setaro can close the engagement, leaving them able to operate, maintain, and evolve it independently.

---

## 1. What we are handing over (scope)

The system is an automated, end-to-end QA pipeline that tests **Mila** (AuPairWorld's chatbot). It consists of four parts that must all transfer together:

| Asset | What it is | Owner today | Owner after handover |
|-------|-----------|-------------|----------------------|
| **Codebase** (~13.3k lines Python, 19 files) | Pipeline, Playwright bridge, tester bot, evaluator, Streamlit dashboard, analytics | Setaro repo | AuPairWorld git repo |
| **Notion workspace** | Test Runs, Test Cases, Executions, Evaluation Pages, Conversations, versioned prompt pages | Setaro Notion | AuPairWorld Notion |
| **Infrastructure** | AWS EC2 + nginx (TLS, basic auth) running 5 long-lived services | Setaro AWS | AuPairWorld cloud |
| **Accounts & secrets** | OpenAI API key, Notion integration token, Mila test login | Setaro accounts | AuPairWorld accounts |

A clean handover means each row's "owner after" column is true **and verified by a test run they trigger themselves**.

---

## 2. Recommended handover approach

I recommend a **phased handover with a defined support/warranty window**, not a one-shot "here are the keys" transfer. The system has real moving parts (a live browser bridge, two LLM accounts, a stateful Notion backend), so the client needs to *operate it under supervision* before we step away.

### Phase 0 — Pre-handover prep (Setaro internal, ~1 week)
- Containerize the app with **Docker Compose** (see §5). This is the single biggest lever for a clean handover — it turns "reproduce this hand-tuned EC2" into "run `docker compose up`."
- Write the missing docs (§8) and scrub all Setaro-specific secrets/accounts from the repo and `.env`.
- Produce a clean `.env.example` with every variable documented and **no real secrets**.

### Phase 1 — Account & infrastructure provisioning (AuPairWorld, with our guidance)
- They create their own OpenAI account, Notion integration, Mila test account, and cloud host (§3).

### Phase 2 — Notion migration (§6)
- Duplicate/transfer the databases into their workspace, re-share to their integration, and re-map the IDs in `.env`.

### Phase 3 — Deploy & parallel run
- Stand up their instance. Run **both** (ours + theirs) against the same test suite for a short period and compare results to prove parity.

### Phase 4 — Supervised operation + training (§8)
- They run real test cycles; we shadow and answer questions. Live training sessions + recorded walkthroughs.

### Phase 5 — Cutover, sign-off, and warranty
- Shut down Setaro's instance, rotate/revoke Setaro credentials, formal acceptance sign-off, then a fixed **30–60 day support window** for bug fixes and questions.

---

## 3. What AuPairWorld will need (software, accounts, infrastructure, access)

### Accounts (they must own these — billing in their name)
- **OpenAI account + API key.** Used by *both* the tester bot and the evaluator, so it drives the running cost. They set their own model (currently `gpt-5-nano`) and spending limits. **This is the one recurring cost and the most common failure mode** (quota exhaustion halts the pipeline).
- **Notion account + internal integration token.** A new integration created in *their* workspace, with every relevant database explicitly shared to it.
- **Mila test account on AuPairWorld staging.** Today the pipeline logs in as `hhc@setaro.de` — a Setaro account. They must supply their own test user (e.g. a dedicated `qa-bot@aupairworld.com`) plus the staging site's HTTP basic-auth credentials. Since they own Mila, this is fully in their control.
- **Cloud host account** (AWS or equivalent) to run the EC2/container host.
- *(Optional)* a **domain/DNS record** if they want a real TLS certificate instead of the current self-signed cert on the dashboard.

### Software / runtime
- Python 3.10+ (or just Docker, if we containerize — strongly recommended)
- Playwright + a headless Chromium browser
- nginx (reverse proxy: TLS, HTTP basic auth, rate limiting)
- Python libs: `streamlit`, `streamlit-autorefresh`, `notion-client`, `openai`, `python-dotenv`, `playwright`

### Access / permissions they'll need internally
- Admin on their OpenAI and Notion workspaces
- SSH / console access to the host
- Git access to the transferred repo
- Edit access to the Notion test databases (operators) + the prompt pages (prompt maintainers)

---

## 4. How they operate & maintain it going forward

The day-to-day operating model is already well-designed and **mostly non-technical** — that's a selling point for the handover.

### Routine operation (non-technical operator)
1. **Trigger a run:** set a row in the Notion *Test Runs* DB to `Prepare Test Run`. The pipeline does the rest.
2. **Monitor:** the Streamlit dashboard (live pipeline status, pass/fail, KPIs, log tail) at `https://<host>/`.
3. **Read results:** evaluation pages and scores appear back in Notion automatically.
4. **Iterate on prompts:** test-case and evaluation prompts are **versioned as Notion child pages** — a non-technical person can add a new version and the pipeline auto-picks the latest. This is a genuinely nice design that lowers the maintenance burden.

The `./testbot` CLI is the operator's toolkit: `status`, `kill-bridge`, `restart`, `tail`, `errors`, `disk`.

### Maintenance (needs a Python-capable person — see risk in §9)
- **Selector maintenance (the main ongoing risk):** the bridge drives Mila's live web widget via CSS selectors. If AuPairWorld changes Mila's chat UI, the bridge can break and needs a developer to update selectors. The codebase already has hardened multi-selector fallbacks and a diagnostics system that exports a selector-success library to make this easier.
- **Dependency updates:** periodic `pip`/Playwright/browser updates.
- **OpenAI credit top-ups & cost monitoring.**
- **Disk hygiene:** a log-retention job (gzip + delete) keeps `logs/` bounded — schedule it via cron (`0 3 * * *`).
- **Crash recovery is largely automatic:** stale evaluation claims self-heal, analytics writes are non-blocking, and failures are now "loud" rather than silent.

### Operating constraints they must understand
- **Strictly serial execution.** A single shared Mila account means tests run one at a time (~3–5 min each). A 50-case suite is hours, not minutes. Parallelism was deliberately removed because one account has no conversation-isolation. *Higher throughput requires multiple Mila test accounts + code changes — a possible future enhancement.*

---

## 5. Deployment / hosting / execution requirements

### Current state
- One AWS EC2 instance running 5 long-lived Python services, started via `start_services.sh`, fronted by nginx (self-signed TLS + basic auth + 30 r/s rate limit). Services:
  - `test_bot_headless.py` (tester API, :8501)
  - `prepare_test_runs.py`, `run_test_executions.py`, `run_test_evaluations.py` (pipeline daemons)
  - `dashboard_app.py` (Streamlit, :8502)

### Recommendations for the handover
1. **Containerize with Docker Compose** (do this before handover). Removes "works on our box" risk, pins the Chromium/Playwright versions, and makes their deploy a one-liner. Highest-value prep task.
2. **Replace the shell-script process management with `systemd` units** (or container restart policies) so services survive reboots and crashes without manual `start_services.sh`.
3. **Right-size the host:** a small/medium instance is plenty (serial workload), but headless Chromium needs ~1–2 GB RAM headroom. Document the recommended instance type.
4. **Use a real TLS cert** (Let's Encrypt via a domain) instead of self-signed if the dashboard is exposed.
5. **Externalize secrets:** move `.env` secrets into the cloud provider's secret manager rather than a plaintext file on disk.

---

## 6. Notion migration plan

This is the fiddliest part because **page/database IDs are baked into `.env`** and **change on migration**.

### Steps
1. **Create their integration:** AuPairWorld creates a new internal Notion integration in their workspace → gets a new `NOTION_API_KEY`.
2. **Move the content.** Two viable methods:
   - **Duplicate to their workspace** (Notion's "Move to" / duplicate across workspaces) — preserves structure and relations best.
   - **Export → import** (Markdown/CSV) — simpler but **breaks inter-database relations** (Test Runs ↔ Executions ↔ Evaluations ↔ Conversations) and loses some property types. *Prefer duplicate/move.*
3. **Re-share every database & parent page to the new integration** (Notion integrations only see explicitly shared content — easy to miss one and get silent failures).
4. **Re-map all IDs in `.env`:** every `*_DB_ID` and `*_PARENT_PAGE_ID` will be new. There are **8 IDs** to update:
   - `TEST_CASES_DB_ID`, `TEST_RUNS_DB_ID`, `TEST_CASE_EXECUTIONS_DB_ID`, `EVALUATION_PAGES_DB_ID`
   - `CONVERSATIONS_PARENT_PAGE_ID`, `TEST_CASE_PROMPT_PARENT_PAGE_ID`, `EVALUATION_PROMPT_PARENT_PAGE_ID`, `NOTION_PARENT_PAGE_ID`
5. **Preserve the prompt-versioning structure:** the prompt parent pages must keep their child-page-per-version layout, since the pipeline auto-selects the most recently edited child.
6. **Verify property names match** what the code expects (e.g. `Evaluation JSON`, `Evaluation score`, `Evaluation Page Link`, and the status values including the project's `Test Execustion Started` typo).
7. **Smoke test:** trigger one `Prepare Test Run` end-to-end in their workspace and confirm a conversation page + evaluation page appear.

> **Decision point:** do they want a *fresh start* (empty databases, keep only schema + prompts) or a *full history migration* (carry over past runs/evaluations)? Fresh start is cleaner and faster; history migration preserves trend KPIs.

---

## 7. Security & credential handover (do not skip)

The current `.env` contains **live Setaro-owned secrets** that must be rotated/revoked, not transferred:
- `OPENAI_API_KEY` — revoke after cutover; they issue their own.
- `NOTION_API_KEY` — Setaro integration; they create their own.
- `MILA_LOGIN_USER=hhc@setaro.de` / `MILA_LOGIN_PASS` — **a Setaro staff account.** Replace with an AuPairWorld-owned QA account and rotate the password.
- `MILA_HTTP_USER` / `MILA_HTTP_PASS` — staging basic-auth; they control these.

Checklist: issue new credentials → deploy their instance with them → verify → **revoke all Setaro credentials** → confirm Setaro instance is shut down. Never commit real secrets to the transferred repo (ship `.env.example` only).

---

## 8. Documentation, training & onboarding

### Already exists (in `docs/`)
- `OPERATOR_GUIDE.md` — day-to-day operations & failure modes
- `ANALYTICS_SCHEMA.md` — analytics event stream spec
- `eval-prompt-v5.md` — current evaluation rubric (9 dimensions)

### To produce for handover
- **This handover plan** + a **System Architecture overview** (we already have the full breakdown from the codebase analysis — package it as a doc/diagram).
- **Deployment runbook** — provision → configure → `docker compose up` → verify, including the Notion ID re-mapping steps.
- **`.env` reference** — every variable, what it does, safe defaults.
- **Troubleshooting / "what breaks & who fixes it"** — selector breakage, OpenAI quota, Mila staging down, stuck executions.
- **Prompt-maintenance guide** — how to add a new prompt version in Notion (for non-technical staff).
- **Cost guide** — expected OpenAI spend per run and how to set limits.

### Training
- **2–3 live knowledge-transfer sessions** (recorded): (1) operator walkthrough, (2) deployment & maintenance for their dev, (3) prompt/KPI tuning.
- A **parallel-run period** (Phase 3) where they drive and we shadow is worth more than any document.

---

## 9. Risks & open decisions (for our discussion before we present)

| # | Risk / question | Why it matters | Recommendation |
|---|-----------------|----------------|----------------|
| 1 | **Does AuPairWorld have a technical/dev person?** | Selector maintenance & pipeline debugging need Python skills. Pure-operator handover is not viable long-term. | Confirm a named technical owner on their side, or offer a **support retainer**. |
| 2 | **OpenAI cost ownership & budget** | It's the recurring cost and the top failure mode. | They own billing; we document expected spend + set hard limits. |
| 3 | **Notion as the production datastore** | Elegant for UX, but rate-limited & fragile at volume; IDs break on migration. | Acceptable for current scale; flag as a known limitation. |
| 4 | **Serial single-account throughput** | Large suites take hours. | Document the limit; offer multi-account parallelism as paid future work. |
| 5 | **Self-signed cert / plaintext `.env` secrets** | Security posture. | Move to real TLS + secret manager during handover. |
| 6 | **Fresh start vs full history migration** | Affects effort & whether trend KPIs survive. | Recommend fresh start unless they need history. |
| 7 | **Where do they host?** | Lift-and-shift EC2 vs redeploy in their cloud. | Containerize so host choice is theirs and low-risk. |
| 8 | **Post-handover support scope** | Avoid open-ended obligations. | Fixed 30–60 day warranty window, then optional paid support. |

---

## 10. Suggested next steps

1. Internal Setaro review of this draft; resolve the §9 decision points.
2. Confirm with AuPairWorld: technical owner, hosting preference, history vs fresh start, support expectations.
3. Turn the agreed approach into a dated, milestone-based handover schedule with acceptance criteria.
4. Execute Phase 0 prep (containerize + docs + secret scrub) so we can demo a clean deploy in the kickoff.
