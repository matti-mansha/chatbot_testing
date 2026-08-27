#!/usr/bin/env python3
"""
preflight.py — validate configuration before starting the pipeline.

Run this after filling in .env and before the first ./start_services.sh:

    ./env/bin/python preflight.py

Why this exists
---------------
Every Notion object must be explicitly shared with the integration. An ID that
is correct but unshared — or an ID that points at an archived database — returns
404 object_not_found. The pipeline does NOT crash on this: it idle-fails every
10 seconds. The Setaro instance ran that way from April to July 2026, producing
no test data and ~850 MB of 404 spam.

This script turns that silent failure into a loud one, before it can start.

Exit codes:  0 = all checks passed (warnings allowed)   1 = at least one failure
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Output helpers — match the pipeline's existing log idiom
# ---------------------------------------------------------------------------

FAILURES: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str) -> None:
    print(f"   \033[32m✅\033[0m {msg}")


def fail(msg: str, hint: str = "") -> None:
    print(f"   \033[31m❌\033[0m {msg}")
    if hint:
        print(f"      → {hint}")
    FAILURES.append(msg)


def warn(msg: str, hint: str = "") -> None:
    print(f"   \033[33m⚠️\033[0m  {msg}")
    if hint:
        print(f"      → {hint}")
    WARNINGS.append(msg)


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ---------------------------------------------------------------------------
# 1. .env presence and required variables
# ---------------------------------------------------------------------------

# Variables with no usable default — the pipeline cannot run without them.
REQUIRED = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "NOTION_API_KEY",
    "TEST_CASES_DB_ID",
    "TEST_RUNS_DB_ID",
    "TEST_CASE_EXECUTIONS_DB_ID",
    "EVALUATION_PAGES_DB_ID",
    "CONVERSATIONS_PARENT_PAGE_ID",
    "TEST_CASE_PROMPT_PARENT_PAGE_ID",
    "EVALUATION_PROMPT_PARENT_PAGE_ID",
    "MILA_URL",
    "MILA_LOGIN_USER",
    "MILA_LOGIN_PASS",
    "MILA_HTTP_USER",
    "MILA_HTTP_PASS",
]

# Read only by archive/ code. Harmless, but they look live and mislead operators.
DEAD_VARS = ["NOTION_PARENT_PAGE_ID", "TEST_BOT_URL", "CHECK_INTERVAL"]

# Notion objects to resolve: (env var, human label)
NOTION_OBJECTS = [
    ("TEST_CASES_DB_ID", "Test cases"),
    ("TEST_RUNS_DB_ID", "Test Runs"),
    ("TEST_CASE_EXECUTIONS_DB_ID", "Test Case Executions"),
    ("EVALUATION_PAGES_DB_ID", "Evaluation Pages"),
    ("CONVERSATIONS_PARENT_PAGE_ID", "Conversations parent"),
    ("TEST_CASE_PROMPT_PARENT_PAGE_ID", "Test Prompts parent"),
    ("EVALUATION_PROMPT_PARENT_PAGE_ID", "Evaluation Prompt parent"),
]


def check_env_file() -> None:
    section("1. Configuration file")

    if not Path(".env").exists():
        fail(".env not found", "cp .env.example .env  — then fill in the values")
        # Nothing further can be checked.
        summarise_and_exit()

    try:
        from dotenv import load_dotenv
    except ImportError:
        fail("python-dotenv not installed", "./env/bin/pip install -r requirements.txt")
        summarise_and_exit()

    load_dotenv()
    ok(".env loaded")

    missing = [v for v in REQUIRED if not (os.getenv(v) or "").strip()]
    if missing:
        for v in missing:
            fail(f"{v} is not set", "see .env.example for what this value should be")
    else:
        ok(f"all {len(REQUIRED)} required variables are set")

    present_dead = [v for v in DEAD_VARS if (os.getenv(v) or "").strip()]
    if present_dead:
        warn(
            f"stale variables present: {', '.join(present_dead)}",
            "only archive/ code reads these — safe to delete from .env",
        )


# ---------------------------------------------------------------------------
# 2. Notion — auth, object resolution, schema
# ---------------------------------------------------------------------------

def check_notion() -> None:
    section("2. Notion")

    try:
        from notion_client import Client as NotionClient
        from notion_client.errors import APIResponseError
    except ImportError:
        fail("notion-client not installed", "./env/bin/pip install -r requirements.txt")
        return

    token = (os.getenv("NOTION_API_KEY") or "").strip()
    if not token:
        fail("NOTION_API_KEY missing — skipping all Notion checks")
        return

    client = NotionClient(auth=token)

    try:
        me = client.users.me()
        ok(f"token authenticates as: {me.get('name') or me.get('id')}")
    except APIResponseError as e:
        fail(f"Notion token rejected ({e.code})", "regenerate the integration token")
        return
    except Exception as e:  # network, TLS, proxy
        fail(f"cannot reach the Notion API: {e}")
        return

    executions_is_db = False

    for var, label in NOTION_OBJECTS:
        obj_id = (os.getenv(var) or "").strip()
        if not obj_id:
            fail(f"{label}: {var} not set")
            continue

        # The codebase is inconsistent about which IDs are databases and which
        # are pages (see the "# Actually page IDs" comment in prepare_test_runs).
        # Try both rather than assuming.
        kind = None
        try:
            client.databases.retrieve(database_id=obj_id)
            kind = "database"
        except Exception:
            try:
                client.pages.retrieve(page_id=obj_id)
                kind = "page"
            except APIResponseError as e:
                if e.code == "object_not_found":
                    fail(
                        f"{label}: 404 object_not_found",
                        "either the ID is wrong/archived, OR the object is not "
                        "shared with this integration (Notion → ••• → Connections)",
                    )
                elif e.code == "unauthorized":
                    fail(f"{label}: unauthorized", "share this object with the integration")
                else:
                    fail(f"{label}: {e.code}")
                continue
            except Exception as e:
                fail(f"{label}: {e}")
                continue

        ok(f"{label} resolves ({kind})")
        if var == "TEST_CASE_EXECUTIONS_DB_ID" and kind == "database":
            executions_is_db = True

    if executions_is_db:
        check_executions_schema(client)


def check_executions_schema(client) -> None:
    """The evaluator writes to named properties; a rename breaks it silently."""
    section("3. Notion schema — Executions database")

    db_id = (os.getenv("TEST_CASE_EXECUTIONS_DB_ID") or "").strip()
    try:
        db = client.databases.retrieve(database_id=db_id)
    except Exception as e:
        warn(f"could not read schema: {e}")
        return

    props = db.get("properties", {})
    lowered = {k.lower(): k for k in props}

    # Property-name checks. The link property is optional (blank = skip linking).
    for env_var, default in [
        ("EVALUATION_TEXT_PROPERTY", "Evaluation JSON"),
        ("EVALUATION_SCORE_PROPERTY", "Evaluation score"),
    ]:
        name = (os.getenv(env_var) or default).strip()
        if name.lower() in lowered:
            ok(f'property "{name}" exists')
        else:
            fail(
                f'property "{name}" not found on the Executions database',
                f"available: {', '.join(sorted(props)) or '(none)'}",
            )

    link_name = (os.getenv("EVALUATION_PAGE_LINK_PROPERTY") or "").strip()
    if not link_name:
        warn("EVALUATION_PAGE_LINK_PROPERTY is blank", "evaluation back-links will be skipped")
    elif link_name.lower() in lowered:
        ok(f'property "{link_name}" exists')
    else:
        fail(f'property "{link_name}" not found on the Executions database')

    # Status option values must match Notion exactly — including the known typo.
    #
    # The status property name is HARDCODED in the pipeline (run_test_executions.py,
    # run_test_evaluations.py) as "Test Execution Status" — it is NOT configurable
    # via .env, so check that exact name rather than a generic "Status".
    status_key = lowered.get("test execution status")
    if not status_key:
        fail(
            'property "Test Execution Status" not found',
            "the pipeline hardcodes this name; the executions DB must have it",
        )
        return

    prop = props[status_key]
    opts = (prop.get("status") or prop.get("select") or {}).get("options", [])
    names = {o.get("name", "") for o in opts}
    if not names:
        warn("Test Execution Status has no options to verify")
        return

    # Values that are configurable via .env ...
    configurable = [
        ("EXECUTION_SUCCESS_STATUS", "Test Executed"),
        ("EXECUTION_FAILED_STATUS", "Test Execustion Started"),
    ]
    # ... and values written as string literals by run_test_evaluations.py.
    hardcoded = ["Evaluation started", "Evaluation completed", "Not started"]

    for env_var, default in configurable:
        val = (os.getenv(env_var) or default).strip()
        if val in names:
            ok(f'status option "{val}" exists')
        else:
            fail(
                f'status option "{val}" does not exist in Notion',
                f"available: {', '.join(sorted(names))}. "
                "Note the project's known typo 'Test Execustion Started' — the "
                ".env value must match Notion exactly, typo included.",
            )

    for val in hardcoded:
        if val in names:
            ok(f'status option "{val}" exists (hardcoded)')
        else:
            fail(
                f'status option "{val}" is missing',
                "written as a literal by run_test_evaluations.py; add it in Notion",
            )

    # "Test Case Status" is also hardcoded, and prepare_test_runs.py maps the
    # value "Inactive" -> "In Active" to match the Notion schema's spelling.
    tcs_key = lowered.get("test case status")
    if not tcs_key:
        warn('property "Test Case Status" not found', "prepare_test_runs.py writes to it")
    else:
        tcs = props[tcs_key]
        tcs_names = {
            o.get("name", "")
            for o in (tcs.get("status") or tcs.get("select") or {}).get("options", [])
        }
        for val in ["Active", "In Active"]:
            if val in tcs_names:
                ok(f'Test Case Status option "{val}" exists')
            else:
                warn(f'Test Case Status option "{val}" is missing')


# ---------------------------------------------------------------------------
# 4. OpenAI
# ---------------------------------------------------------------------------

def check_openai() -> None:
    section("4. OpenAI")

    try:
        from openai import OpenAI
    except ImportError:
        fail("openai not installed", "./env/bin/pip install -r requirements.txt")
        return

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()

    if not key:
        fail("OPENAI_API_KEY missing — skipping OpenAI checks")
        return
    if not model:
        fail("OPENAI_MODEL is not set", "the code has no default; the pipeline will not start")
        return

    client = OpenAI(api_key=key, timeout=30)

    try:
        available = {m.id for m in client.models.list()}
        ok("API key accepted")
    except Exception as e:
        fail(f"OpenAI rejected the key or is unreachable: {e}")
        return

    if model in available:
        ok(f'model "{model}" is available to this account')
    else:
        # models.list() does not always enumerate every usable alias.
        warn(
            f'model "{model}" not listed for this account',
            "verify the name, and confirm billing/quota is active",
        )


# ---------------------------------------------------------------------------
# 5. Playwright / Chromium
# ---------------------------------------------------------------------------

def check_playwright() -> None:
    section("5. Playwright / Chromium")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        fail("playwright not installed", "./env/bin/pip install -r requirements.txt")
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        ok("headless Chromium launches")
    except Exception as e:
        fail(
            f"Chromium failed to launch: {e}",
            "./env/bin/playwright install chromium && "
            "./env/bin/playwright install-deps",
        )


# ---------------------------------------------------------------------------
# 6. MILA reachability
# ---------------------------------------------------------------------------

def check_mila() -> None:
    section("6. MILA staging site")

    try:
        import httpx
    except ImportError:
        fail("httpx not installed", "./env/bin/pip install -r requirements.txt")
        return

    url = (os.getenv("MILA_URL") or "").strip()
    if not url:
        fail("MILA_URL not set — skipping")
        return

    user = (os.getenv("MILA_HTTP_USER") or "").strip()
    pw = (os.getenv("MILA_HTTP_PASS") or "").strip()
    auth = (user, pw) if user else None

    try:
        r = httpx.get(url, auth=auth, timeout=20, follow_redirects=True)
    except Exception as e:
        fail(f"cannot reach {url}: {e}", "check the VM's network/DNS and any egress firewall")
        return

    if r.status_code == 401:
        fail("401 Unauthorized", "MILA_HTTP_USER / MILA_HTTP_PASS are wrong or missing")
    elif r.status_code >= 400:
        fail(f"HTTP {r.status_code} from {url}")
    else:
        ok(f"reachable (HTTP {r.status_code})")
        warn(
            "this only proves the page loads",
            "it does NOT prove the chat widget selectors still match — "
            "confirm with a real test run",
        )


# ---------------------------------------------------------------------------

def summarise_and_exit() -> None:
    print()
    if FAILURES:
        print(f"\033[31m\033[1m{len(FAILURES)} check(s) FAILED\033[0m — do not start the pipeline yet:")
        for f in FAILURES:
            print(f"   • {f}")
        if WARNINGS:
            print(f"\n{len(WARNINGS)} warning(s).")
        print()
        sys.exit(1)

    if WARNINGS:
        print(f"\033[33m\033[1mAll checks passed with {len(WARNINGS)} warning(s):\033[0m")
        for w in WARNINGS:
            print(f"   • {w}")
    else:
        print("\033[32m\033[1mAll checks passed.\033[0m")

    print("\nNext:  ./start_services.sh   then   ./testbot status\n")
    sys.exit(0)


def main() -> None:
    print("\n\033[1mMILA pipeline preflight\033[0m")
    print("Validating configuration before startup.")

    check_env_file()
    check_notion()
    check_openai()
    check_playwright()
    check_mila()
    summarise_and_exit()


if __name__ == "__main__":
    main()
