# MILA Chatbot Testing Framework — Installation Runbook

**Target:** AuPairWorld test VM (`apw-test`, Ubuntu, VMware)
**Notion workspace:** AuPairWorld → *Chatbot Handover*
**Date:** 2026-08-24
**Prepared by:** Setaro GmbH

---

## Step 1 — Codebase audit

Static analysis plus evidence from the live run on the client VM.
Ranked by operational impact.

### P1 — Blocking

| # | Issue | Location | Effect |
|---|---|---|---|
| 1 | **Login success is never verified** | `playwright_bridge_bot_headless.py:1001` | Logs `✓ Login attempt complete` immediately after clicking, regardless of outcome. Bad credentials surface 30s later as an unrelated chat-widget timeout. **This caused the failed demo run.** |
| 2 | **Page notices intercept the chat button** | `open_mila_chat()` | A Drupal `<qz-message status="error" lifespan="0">` banner overlays the FAB and absorbs every click until timeout. The selector itself is correct — this is not selector drift. |

### P1 — Found and fixed during this install

| # | Issue | Location | Effect |
|---|---|---|---|
| 3 | **`Test Execution Status` never set on new rows** | `prepare_test_runs.py` | Relied on the Notion column default, which does not survive workspace duplication. Rows were created with a null status; `run_test_executions.py:280` filters for `"Not started"` and can never match. Pipeline polls forever, logs no error, does nothing. Verified by replaying the executor's filter against the live DB: **0 rows matched**. |

### P2 — Will affect the client

| # | Issue | Location | Effect |
|---|---|---|---|
| 4 | **Tester API is unauthenticated on all interfaces** | `test_bot_headless.py:647` | Binds `0.0.0.0:8501` with no auth. Anyone on the client LAN can drive it and consume their OpenAI budget. |
| 5 | **Undeclared dependencies** | `requirements.txt` | `httpx` and `pydantic` are top-level imports in 5 modules but were satisfied only transitively via `openai`. Fixed. |
| 6 | **Secret files not ignored** | `.gitignore` | Matched `.env` exactly; `.env copy` and `.env-prod` (4 live secrets each) were committable. Fixed to `.env*` with `!.env.example`. |
| 7 | **Config errors degrade silently** | `prepare_test_runs.py:85` | `find_database_in_page()` swallows exceptions and returns the page ID, turning a clear config error into a downstream 404. |

### P3 — Minor

| # | Issue | Location |
|---|---|---|
| 8 | `mkdir` without `parents=True` — a nested `LOG_DIR` crashes at startup | `logging_config.py:43` |
| 9 | Probes the EC2 metadata endpoint on a non-EC2 host (2s stall, prints `<ec2-public-ip>`) | `start_services.sh` |
| 10 | Dead variables read only by `archive/`: `NOTION_PARENT_PAGE_ID`, `TEST_BOT_URL`, `CHECK_INTERVAL` | `.env` |
| 11 | `EVALUATION_PAGE_LINK_PROPERTY` is set, but no such property exists in the AuPairWorld schema — back-links are silently skipped | `.env` / Notion |

### Verified sound — no action required

- **All 40 `httpx` calls carry explicit timeouts.** No hang risk.
- **Analytics writes use `fcntl.LOCK_EX`** — safe across the bridge subprocess and parent services.
- **Bare `except:` blocks are confined to error-logging paths** — correct practice, not defects.
- **Browser cleanup** is covered by the `with sync_playwright()` context manager on exception paths.
- **Notion property matching is case-insensitive**, so `Evaluation score` vs `Evaluation Score` is a non-issue.

### Status

- Items 3, 5, 6 — fixed in the working tree, **not yet deployed to the client VM**.
- Items 1, 2 — proposed, not yet implemented.
- Items 4, 7–11 — documented, no change made.

---

## Step 2 — Fix P1 items 1 and 2

**File:** `playwright_bridge_bot_headless.py`

### Fix 1 — Verify that login actually succeeded

*Before:* `✓ Login attempt complete` was logged unconditionally after clicking the
button. A rejected password produced a success message, and the real failure only
appeared ~30s later as a chat-widget timeout — pointing at the wrong component.

*After:* added `verify_mila_login()`, called on both the "form submitted" and the
"no form found (already logged in?)" paths. It fails loudly using two independent
signals:

1. A Drupal error notice on the page — reported with **the site's own wording**
   (e.g. *"Unrecognized username or password"*, or a flood-control block).
2. Still on `/user/login` after submitting.

The `RuntimeError` it raises is re-raised explicitly, so the existing
`except Exception` handler cannot log-and-continue past an auth failure.

### Fix 2 — Stop page notices from blocking the chat button

*Before:* `fab.click()` with the default 30s timeout. A Drupal notice banner
(`<qz-message lifespan="0">`, never auto-dismisses) is laid out over the FAB and
absorbs every click. Playwright reported only
`<qz-message ...> intercepts pointer events`, which reads like selector drift.
**The selector was always correct.**

*After:*
- `dismiss_page_notices()` removes notice banners before clicking.
- The click uses a **5s** timeout instead of 30s, so it fails fast.
- On failure it sweeps notices again and falls back to a JS click, which is not
  subject to pointer-event interception at all.

### Verification

Added `tools/test_bridge_overlay_fixes.py` — reproduces the production failure on
a synthetic page and asserts both fixes. Needs only playwright + chromium; makes
no network calls and never touches the live MILA site.

```
./env/bin/python tools/test_bridge_overlay_fixes.py
```

| Test | Result |
|---|---|
| 1. Reproduce production failure — plain click intercepted | PASS (`qz-message intercepts pointer events`) |
| 2. Fix 2 — banner removed, click succeeds | PASS |
| 3. Fix 2 — JS click works even with the banner present | PASS |
| 4. Fix 1 — login error detected, text captured verbatim | PASS |
| 5. Fix 1 — clean page yields no false positive | PASS |

Test 1 reproduces the *exact* error string seen on the client VM, so the fixes are
verified against the real failure rather than an assumed one.

### Status

Both fixes are in the working tree and tested. **Not yet deployed to the client VM.**

---

## Step 3 — Investigate the login failure

**Hypothesis tested:** the MILA login page's CSS/layout changed, so the bridge
could not detect the form fields.

**Result: hypothesis disproven.** The login form was fetched directly from
staging (`HTTP 200`, 75,853 bytes) and its markup matches the bridge exactly:

| Element | Live markup | Bridge selector | Match |
|---|---|---|---|
| Username | `id="edit-name"`, label *"Login by username/email address"* | `get_by_label("Login by username/email address")` | yes |
| Password | `id="edit-pass"`, label *"Password"* | `get_by_label("Password")` | yes |
| Submit | `<input type="submit" value="Log in">` | `get_by_role("button", name="Log in")` | yes |

Corroborated by the bridge's own log from the failed run — all three elements were
found on the first attempt, with no fallback selectors triggered:

```
✓ Username field found
✓ Password field found
➡️ Clicking Log in button
```

**Conclusion:** field detection is not the problem. The form submits correctly and
the site rejects the credentials afterwards. The failure is authentication, not
automation.

**Remaining candidates, in order of likelihood:**

1. **Drupal flood control** — the account locks after repeated failed logins and
   then rejects even the correct password. Clears on its own, or by clearing the
   flood table. Given the number of runs today, this is a live possibility.
2. **Wrong password** — `MILA_LOGIN_PASS` does not match the account.
3. Account blocked or not activated.

**Note:** the login page sits behind HTTP basic auth (`apw`). A browser hitting it
without those credentials receives `401` and a blank page — not a site error.

**Note:** `form_build_id` is regenerated per request, so any scripted login must
fetch the page first and reuse the cookie jar.

### Status

Diagnosis complete. Credential verification must be run by the operator, since it
requires the account password. Once Step 2's fix is deployed, the bridge reports
the site's own rejection reason directly in the log — no manual check needed.

---

## Step 3b — Correction: the login was NOT failing

Evidence from the failed run's diagnostic report (retrieved from the Notion
conversation page) and from a successful manual browser login.

**Network status codes during the failed run:**

```
200: 385    204: 4    303: 1    403: 1    404: 1
```

The **303** is decisive. Drupal answers a *successful* login with a 303 redirect;
a rejected login re-renders the form as 200. Combined with:

- the chat container being present afterwards (`chat-collapsed`), and
- an operator logging in manually with the same account without difficulty,

**the bridge's login succeeded.** The `qz-message status="error"` banner on the
page was unrelated to authentication.

Also captured in the same run: **4 JavaScript errors** and one **403** response.

```
1. Cannot read properties of null (reading 'removeAttribute')
2. Cannot read properties of null (reading 'addEventListener')
3. Identifier 'dl' has already been declared
4. Cannot read properties of null (reading 'addEventListener')
```

### Consequence: Step 2's Fix 1 was wrong and has been corrected

As first written, `verify_mila_login()` raised on *any* error notice. On this very
page — successful login, unrelated error banner — it would have aborted a working
run, replacing a confusing failure with a false one.

**Corrected logic**, in order of authority:

1. An error notice matching a known authentication rejection → **fail**.
2. Still on `/user/login` after submitting → **fail**.
3. Any other error notice → **warn only**, do not fail. Those banners are the
   business of `dismiss_page_notices()`.

Matching uses `AUTH_FAILURE_PHRASES`. A test caught that the original
`"too many failed login"` phrase does **not** match Drupal's real flood-control
wording (*"There have been more than 5 failed login attempts…"*); it now matches
the invariant substring `"failed login attempts"`.

### Revised conclusion

The real and only blocker is the **error banner overlaying the chat FAB** —
precisely what Fix 2 addresses. Fix 2 is unchanged and remains correct.

### Test coverage

`tools/test_bridge_overlay_fixes.py` — 13 checks, all passing. Tests 6 and 7 are
regression guards for this correction:

| Test | Result |
|---|---|
| 6. Auth-rejection wording *is* treated as login failure (incl. flood control) | PASS |
| 7. Unrelated site errors are *not* treated as login failures | PASS |

### Outstanding

The exact text of the blocking banner is still unknown. It is captured on the
client VM at `diagnostics/20260824_153907_dom_snapshots.json`.

---

## Step 4 — Local verification against live staging

Both fixes were run against the real MILA staging site using the actual bridge
functions (no OpenAI, no Notion, no conversation — load → cookies → login → open
chat).

### Run A — with the wrong password (`.env`, 12 characters)

```
17:50:05  ➡️ Clicking Log in button
17:50:11  ❌ LOGIN FAILED — the site rejected the credentials
          Site said: Unrecognized username or password. Forgot your password?
          post-login url = https://stage.aupairworld.com/en/user/login
```

**Fix 1 verified.** Failure reported in **6 seconds**, quoting the site's own
wording, instead of surfacing 30s later as an unrelated chat-widget timeout. Both
signals agreed (auth phrase matched + still on `/user/login`).

### Run B — with the correct password (`.env-prod`, 9 characters)

```
🔓 Login complete and verified.
post-login url = https://stage.aupairworld.com/en/user/24?check_logged_in=1
notices after login: (none)
✓ Mila chat widget opened     ->  chat open = True
```

**No false positive** from Fix 1, and the chat widget opened successfully.

### Correction to Step 3b

Step 3b concluded from a single `303` in the network summary that the client VM's
login had succeeded. **That was an over-reading.** Run A shows the same account
failing with precisely the `qz-message status="error"` banner seen on the VM, and
Run B shows the chat container reporting `chat-collapsed` **even when logged out** —
so its presence never evidenced a successful login. The original Step 3 diagnosis
(an authentication problem) was the correct one.

The narrowing applied to Fix 1 in Step 3b remains correct and is retained: Run A
proves it still catches genuine auth failures, Run B proves it does not fire on a
healthy session.

### Key finding

`MILA_LOGIN_PASS` on the client VM is **already correct** — it matches `.env-prod`
and authenticates successfully. The VM's failure was therefore *not* a wrong
password.

Most probable cause: **Drupal flood control**. Repeated failed attempts during the
day temporarily block the account, and Drupal then rejects even the correct
password — producing exactly the observed error banner. This clears on its own.

Unconfirmed, and only confirmable by re-running on the VM. With Step 2's fixes
deployed, any repeat states the reason outright in the log rather than presenting
as a widget timeout.

### Note

The repo-local `.env` carries an **outdated 12-character password** and will fail.
`.env-prod` holds the working value.

### Coverage caveat

Fix 2's banner-removal path was **not** exercised against a real banner in Run B —
with valid credentials no banner appears. It remains verified synthetically by
`tools/test_bridge_overlay_fixes.py`.

---

## Step 5 — Decommission the Setaro EC2

**AWS account:** `867344437226` (profile `tabtila-prod`, `eu-central-1`)

The original Setaro-hosted instance was decommissioned. A full snapshot was taken
first, since no backup of any kind existed beforehand.

### Sequence

| # | Action | Result |
|---|---|---|
| 1 | Stop `i-003358d83e91eddd0` | `stopped` — 16:06:19 GMT |
| 2 | Snapshot `vol-09d7dc0ac8d5fccb4` | `snap-049dd700626a4e82b`, 25 GB, **completed** |
| 3 | Terminate `i-003358d83e91eddd0` | `terminated` |
| 4 | Release EIP `eipalloc-027492a59181cc4a3` | `3.122.134.15` released |

### Final state

```
instance i-003358d83e91eddd0   terminated
volume   vol-09d7dc0ac8d5fccb4 deleted (DeleteOnTermination=True)
EIP      3.122.134.15          released — NOT recoverable
snapshot snap-049dd700626a4e82b  completed, 25 GB   <-- sole surviving copy
```

### The snapshot is now the only copy

`snap-049dd700626a4e82b` (`Name: aupairworld-chatbot-testing-final`) holds the
entire root volume, including:

- `logs/analytics/events_*.jsonl` — the vendor's system of record
- nginx configuration and TLS certificates
- all historical diagnostics and service logs

To recover data from it: create a volume from the snapshot in `eu-central-1`,
attach it to any instance, and mount it. **Do not delete this snapshot** until the
AuPairWorld VM is verified end-to-end and any required analytics history has been
exported. Storage cost is roughly **$1.25/month**.

### Documentation now stale

`3.122.134.15` is released and cannot be reclaimed. Every reference to it is dead
and must be corrected before the docs go to the client:

- `docs/OPERATOR_GUIDE.md` — dashboard URL, `/analytics/`, `/health`, the SSH host
  line, and the `certbot` upgrade note
- Any AuPairWorld bookmarks pointing at that address

Note: the dashboard basic-auth password recorded at `docs/OPERATOR_GUIDE.md:12`
now protects nothing, but **should still be treated as compromised and rotated**
if reused anywhere else.

### Cost

Previously ~$17/month (t2.small) plus ~$3.60 EIP. Now ~$1.25/month for snapshot
storage alone.
