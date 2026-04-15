# run_test_executions.py
"""
Continuous Test Execution Runner
Monitors for test cases with status 'Not started' and executes them automatically.
Runs in a loop like prepare_test_runs.py
"""
import os
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import subprocess

from dotenv import load_dotenv
import httpx

# Import the improved formatter
from format_conversation_page import create_formatted_conversation_page
from logging_config import setup_logging, log_exception, log_api_call

# =====================================
# LOAD .env
# =====================================

BASE_DIR = pathlib.Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# Set up logging
logger = setup_logging("test_execution")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
TEST_CASE_EXECUTIONS_PAGE_ID = os.getenv("TEST_CASE_EXECUTIONS_DB_ID")

# Conversations parent page
CONVERSATIONS_PARENT_PAGE_ID = os.getenv("CONVERSATIONS_PARENT_PAGE_ID")

# Playwright bridge script
BRIDGE_SCRIPT = os.getenv("BRIDGE_SCRIPT", "playwright_bridge_bot_headless.py")

# ✅ NEW: Check interval configuration
CHECK_INTERVAL = int(os.getenv("EXECUTION_CHECK_INTERVAL", "10"))  # seconds

# Status names used when updating the "Test Execution Status" field in Notion.
# The evaluator queries strictly for `EXECUTION_SUCCESS_STATUS`, so anything else
# is effectively "not ready for evaluation" — which is what we want for failures.
#
# By default, a failed bridge run leaves the row in "Test Execustion Started"
# (matching the project's existing typo) so no Notion schema change is required.
# If you add a "Test Execution Failed" option to the Notion DB, point this env
# var at it for cleaner reporting.
EXECUTION_SUCCESS_STATUS = os.getenv("EXECUTION_SUCCESS_STATUS", "Test Executed")
EXECUTION_FAILED_STATUS = os.getenv("EXECUTION_FAILED_STATUS", "Test Execustion Started")

# How many bridge subprocesses can run concurrently.
#
# Default is 1 (serial). Per-bridge client state is fully isolated
# (separate Playwright browser context, separate cookie jar, separate
# login → separate Drupal session cookie, separate per-context
# sessionStorage for the widget's client-side chat history), so the
# browser side is parallel-safe.
#
# The reason for the conservative default is the SERVER side. Mila's
# xi-deepchat API request body contains no conversation_id or thread_id
# — only assistant_id and the hardcoded contexts.current_route — so
# conversation state on the backend must be keyed by something derived
# from the request (session cookie or Drupal user_id). Without being
# able to read the Drupal module source, we can't prove the backend
# isn't keyed on user_id, in which case parallel workers using the same
# MILA account would all share/corrupt each other's conversation thread.
#
# To safely use EXECUTION_PARALLELISM > 1, use a pool of distinct MILA
# accounts (one per worker). Until account pooling is implemented here,
# set this to 1. You can still override via env var for experiments.
EXECUTION_PARALLELISM = int(os.getenv("EXECUTION_PARALLELISM", "1"))

# Notion API
NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

logger.info("✓ Starting Test Case Executions Runner (Continuous Mode)")
logger.info(f"Bridge Script: {BRIDGE_SCRIPT}")
logger.info(f"Conversations Parent: {CONVERSATIONS_PARENT_PAGE_ID}")
logger.info(f"Check Interval: {CHECK_INTERVAL}s")
print("✓ Starting Test Case Executions Runner (Continuous Mode)")


# =====================================
# HELPER: ID & URL
# =====================================

def format_notion_id(notion_id: str) -> str:
    """Convert a Notion ID to UUID format with hyphens if it is 32 chars."""
    clean_id = (notion_id or "").replace("-", "")
    if len(clean_id) == 32:
        return f"{clean_id[0:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:32]}"
    return notion_id


def notion_page_url(page_id: str) -> str:
    """Build the web URL for a Notion page ID."""
    clean = (page_id or "").replace("-", "")
    return f"https://www.notion.so/{clean}"


def find_database_in_page(page_id: str) -> Optional[str]:
    """
    Given a page ID, try to find a child database in its blocks.
    If none found or call fails, fall back to assuming the ID is already a database ID.
    """
    if not page_id:
        return None

    logger.debug(f"Finding database in page: {page_id}")
    page_id_uuid = format_notion_id(page_id)

    try:
        start_time = time.time()
        resp = httpx.get(
            f"{NOTION_API_BASE}/blocks/{page_id_uuid}/children",
            headers=HEADERS,
            timeout=30.0,
        )
        duration = time.time() - start_time
        log_api_call(logger, "GET", f"{NOTION_API_BASE}/blocks/{page_id_uuid}/children", 
                    resp.status_code, duration)
        
        resp.raise_for_status()
        data = resp.json()

        for block in data.get("results", []):
            btype = block.get("type")
            if btype in ["child_database", "table"]:
                db_id = block.get("id")
                logger.info(f"✓ Found {btype} in page: {db_id}")
                return db_id

        logger.debug("No child database found, using page ID as database ID")
        return page_id_uuid
        
    except Exception as e:
        log_exception(logger, e, "find_database_in_page")
        logger.warning(f"⚠️ Error finding database in page {page_id}: {e}")
        return page_id_uuid


# =====================================
# NOTION HELPERS
# =====================================

def extract_property_value(page: Dict[str, Any], property_name: str) -> Any:
    """Extract value from a Notion property based on its type."""
    try:
        prop = page["properties"].get(property_name, {})
        prop_type = prop.get("type")

        if prop_type == "title":
            lst = prop.get("title", [])
            return lst[0]["plain_text"] if lst else ""
        elif prop_type == "rich_text":
            lst = prop.get("rich_text", [])
            if not lst:
                return ""
            full_text = "".join(item.get("plain_text", "") for item in lst)
            logger.debug(f"Extracted rich_text '{property_name}': {len(full_text)} chars from {len(lst)} chunks")
            return full_text
        elif prop_type == "number":
            return prop.get("number")
        elif prop_type == "select":
            sel = prop.get("select")
            return sel["name"] if sel else None
        elif prop_type == "status":
            st = prop.get("status")
            return st["name"] if st else None
        elif prop_type == "relation":
            return prop.get("relation", [])
        elif prop_type == "date":
            d = prop.get("date")
            return d["start"] if d else None
        else:
            return None
    except Exception as e:
        logger.warning(f"⚠️ Error extracting property '{property_name}': {e}")
        print(f"⚠️ Error extracting property '{property_name}': {e}")
        return None


def update_execution_row_with_results(
    execution_page_id: str,
    status: str,
    duration_seconds: float,
    conversation_page_id: Optional[str],
    number_of_turns: Optional[int] = None,
):
    """
    Update a Test Case Execution row with execution status, duration, date, turns, and conversation link.
    """
    logger.info(f"Updating execution row: {execution_page_id}")
    logger.debug(f"  Status: {status}")
    logger.debug(f"  Duration: {duration_seconds:.2f}s")
    logger.debug(f"  Number of turns: {number_of_turns}")
    logger.debug(f"  Conversation page: {conversation_page_id}")
    
    properties: Dict[str, Any] = {
        "Test Execution Status": {
            "status": {"name": status}
        },
        "Execution duration": {
            "number": duration_seconds,
        },
        "Execution date / time": {
            "date": {"start": datetime.now().isoformat()}
        },
    }
    
    if number_of_turns is not None:
        properties["Number of Turns"] = {
            "number": number_of_turns
        }
        logger.debug(f"  Added Number of Turns: {number_of_turns}")

    if conversation_page_id:
        url = notion_page_url(conversation_page_id)
        properties["Conversation flow"] = {
            "rich_text": [
                {
                    "type": "text",
                    "text": {
                        "content": "Open conversation transcript",
                        "link": {"url": url},
                    },
                }
            ]
        }

    try:
        start_time = time.time()
        resp = httpx.patch(
            f"{NOTION_API_BASE}/pages/{execution_page_id}",
            headers=HEADERS,
            json={"properties": properties},
            timeout=30.0,
        )
        duration = time.time() - start_time
        log_api_call(logger, "PATCH", f"{NOTION_API_BASE}/pages/{execution_page_id}", 
                    resp.status_code, duration)
        
        resp.raise_for_status()
        logger.info(f"   ✓ Updated execution row {execution_page_id} → {status}")
        if number_of_turns is not None:
            logger.info(f"     • Number of Turns: {number_of_turns}")
            print(f"   ✓ Updated execution row → {status} (Turns: {number_of_turns})")
        else:
            print(f"   ✓ Updated execution row → {status}")
        
    except Exception as e:
        log_exception(logger, e, "update_execution_row_with_results")
        logger.error(f"❌ Error updating execution row {execution_page_id}: {e}")
        print(f"❌ Error updating execution row {execution_page_id}: {e}")
        try:
            print("   Response:", resp.text)
        except Exception:
            pass


def get_pending_executions(db_id: str) -> List[Dict[str, Any]]:
    """
    Query Test Case Executions DB for rows where:
      - Test Case Status = Active
      - Test Execution Status = Not started

    Results are sorted by created_time DESCENDING so the newest test run's
    executions are processed first. This is crucial when multiple test runs
    have been queued: without this sort, Notion returns results in
    insertion order (oldest first), which causes a newly-triggered test run
    to sit behind stale executions from earlier runs (possibly days old
    and no longer relevant). With newest-first, triggering a new test run
    immediately jumps its executions to the front of the queue.

    Sort order within a single run is reverse creation order, which is
    functionally fine — tests don't have inter-execution dependencies.
    """
    logger.info(f"Querying for pending executions in database: {db_id}")
    executions: List[Dict[str, Any]] = []
    cursor = None
    page_count = 0

    while True:
        body: Dict[str, Any] = {
            "filter": {
                "and": [
                    {
                        "property": "Test Case Status",
                        "status": {"equals": "Active"},
                    },
                    {
                        "property": "Test Execution Status",
                        "status": {"equals": "Not started"},
                    },
                ]
            },
            "sorts": [
                {
                    "timestamp": "created_time",
                    "direction": "descending",
                }
            ],
        }
        if cursor:
            body["start_cursor"] = cursor

        try:
            start_time = time.time()
            resp = httpx.post(
                f"{NOTION_API_BASE}/databases/{db_id}/query",
                headers=HEADERS,
                json=body,
                timeout=30.0,
            )
            duration = time.time() - start_time
            log_api_call(logger, "POST", f"{NOTION_API_BASE}/databases/{db_id}/query", 
                        resp.status_code, duration)
            
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            executions.extend(results)
            page_count += 1
            
            logger.debug(f"  Page {page_count}: Found {len(results)} executions")

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            
        except Exception as e:
            log_exception(logger, e, "get_pending_executions")
            logger.error(f"❌ Error querying Test Case Executions DB: {e}")
            print(f"❌ Error querying Test Case Executions DB: {e}")
            try:
                print("   Response:", resp.text)
            except Exception:
                pass
            break

    logger.info(f"✓ Found {len(executions)} pending executions (across {page_count} pages)")
    return executions


# =====================================
# PLAYWRIGHT RUNNER
# =====================================

def extract_turn_count_from_output(output: str) -> Optional[int]:
    """Extract number of turns from the bridge script output."""
    logger.debug("Extracting turn count from output")
    
    import re
    
    # Try pattern: "Number of turns: X"
    match = re.search(r"Number of turns:\s*(\d+)", output, re.IGNORECASE)
    if match:
        turns = int(match.group(1))
        logger.debug(f"  Found turn count via 'Number of turns': {turns}")
        return turns
    
    # Try pattern: "Completed X turns"
    match = re.search(r"Completed\s+(\d+)\s+turns", output, re.IGNORECASE)
    if match:
        turns = int(match.group(1))
        logger.debug(f"  Found turn count via 'Completed X turns': {turns}")
        return turns
    
    # Try pattern: "Total turns: X"
    match = re.search(r"Total turns:\s*(\d+)", output, re.IGNORECASE)
    if match:
        turns = int(match.group(1))
        logger.debug(f"  Found turn count via 'Total turns': {turns}")
        return turns
    
    logger.debug("  No turn count found in output")
    return None


def extract_per_turn_scores_from_output(output: str) -> Optional[List[int]]:
    """Extract per-turn completeness scores from bridge script output."""
    import re
    import json as _json

    match = re.search(r"Per-turn scores:\s*(\[[\d,\s]+\])", output)
    if match:
        try:
            scores = _json.loads(match.group(1))
            logger.debug(f"  Extracted per-turn scores: {scores}")
            return scores
        except Exception:
            pass

    logger.debug("  No per-turn scores found in output")
    return None


def run_playwright_bridge(
    test_case_name: str = "",
    persona: str = "",
    test_case_details: str = "",
    test_case_prompt: str = "",
) -> Tuple[str, Optional[int], Optional[List[int]]]:
    """
    Run playwright_bridge_bot.py as a subprocess and capture stdout as conversation transcript.
    Returns: (conversation_text, number_of_turns, per_turn_scores)
    """
    logger.info(f"Running Playwright bridge script: {BRIDGE_SCRIPT}")
    logger.debug(f"  Test case: {test_case_name}")
    logger.debug(f"  Persona: {persona}")
    logger.debug(f"  Details length: {len(test_case_details)} chars")
    logger.debug(f"  Prompt length: {len(test_case_prompt)} chars")
    
    # ✅ FIX: Use virtual environment Python
    python_path = BASE_DIR / "env" / "bin" / "python"
    if not python_path.exists():
        python_path = "python3"
        logger.warning("Virtual environment Python not found, using system python3")
    else:
        python_path = str(python_path)
        logger.debug(f"Using virtual environment Python: {python_path}")
    
    cmd = [
        python_path,  # ✅ NOW USES VENV PYTHON
        BRIDGE_SCRIPT,
        test_case_name or "",
        persona or "",
        test_case_details or "",
        test_case_prompt or "",
    ]
    
    print(f"   ▶ Running: python {BRIDGE_SCRIPT}")
    print(f"   📝 Test case: {test_case_name}")
    print(f"   👤 Persona: {persona}")

    try:
        start_time = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        duration = time.time() - start_time

        bridge_failed = proc.returncode != 0
        logger.info(f"✓ Bridge script completed in {duration:.1f}s (exit code: {proc.returncode})")

        if bridge_failed:
            logger.error(f"   ❌ Bridge script exited with code {proc.returncode}")
            logger.debug(f"   stderr:\n{proc.stderr}")
            print(f"   ❌ Bridge script exited with code {proc.returncode}")
            print("   stderr:\n", proc.stderr)
        else:
            print("   ✓ Bridge script completed")

        conversation_text = proc.stdout.strip()
        if not conversation_text:
            conversation_text = f"(No stdout captured. stderr was:)\n\n{proc.stderr}"
            logger.warning("No stdout captured from bridge script")

        # When the bridge failed, prefix the captured transcript with a clear
        # failure banner so the conversation page we create for debugging is
        # obviously not a normal run.
        if bridge_failed:
            failure_banner = (
                "⚠️ BRIDGE FAILURE\n"
                f"Exit code: {proc.returncode}\n"
                f"Duration: {duration:.1f}s\n"
                f"stderr (tail):\n{(proc.stderr or '')[-2000:]}\n"
                "--- Partial transcript below (may be empty) ---\n\n"
            )
            conversation_text = failure_banner + conversation_text

        logger.debug(f"Captured conversation: {len(conversation_text)} chars")

        number_of_turns = extract_turn_count_from_output(conversation_text)
        if number_of_turns is not None:
            logger.info(f"✓ Extracted turn count: {number_of_turns}")
            print(f"   ✓ Extracted turn count: {number_of_turns}")
        else:
            logger.warning("⚠️ Could not extract turn count from output")
            print("   ⚠️ Could not extract turn count from output")

        per_turn_scores = extract_per_turn_scores_from_output(conversation_text)
        if per_turn_scores is not None:
            logger.info(f"✓ Extracted per-turn scores: {per_turn_scores}")
            print(f"   ✓ Per-turn scores: {per_turn_scores}")

        return conversation_text, number_of_turns, per_turn_scores, bridge_failed

    except Exception as e:
        log_exception(logger, e, "run_playwright_bridge")
        logger.error(f"   ❌ Error running bridge script: {e}")
        print(f"   ❌ Error running bridge script: {e}")
        return f"(Bridge error: {e})", None, None, True


# =====================================
# SINGLE EXECUTION PROCESSOR
# =====================================

def process_single_execution(execution: Dict[str, Any], test_exec_db_id: str) -> bool:
    """
    Process a single test execution.
    Returns True if successful, False otherwise.
    """
    page_id = execution.get("id")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"▶ Processing execution {page_id}")
    logger.info("=" * 60)
    print("\n" + "=" * 60)
    print(f"▶ Processing execution {page_id}")
    print("=" * 60)

    test_case_name = extract_property_value(execution, "Test case name") or ""
    persona = extract_property_value(execution, "Persona") or ""
    test_case_details = extract_property_value(execution, "Test Case Details") or ""
    test_case_prompt = extract_property_value(execution, "Test Case Prompt") or ""
    run_number = extract_property_value(execution, "Test run number") or ""

    logger.info(f"   Test case name: {test_case_name}")
    logger.info(f"   Persona: {persona}")
    logger.info(f"   Test run number: {run_number}")
    logger.info(f"   📝 Test case prompt length: {len(test_case_prompt)} chars")
    logger.info(f"   📋 Test case details length: {len(test_case_details)} chars")
    print(f"   Test case name: {test_case_name}")
    print(f"   Persona: {persona}")
    print(f"   Test run number: {run_number}")
    print(f"   📝 Prompt: {len(test_case_prompt)} chars")
    print(f"   📋 Details: {len(test_case_details)} chars")

    # Mark as "Test Execution Started"
    try:
        start_time = time.time()
        resp = httpx.patch(
            f"{NOTION_API_BASE}/pages/{page_id}",
            headers=HEADERS,
            json={
                "properties": {
                    "Test Execution Status": {
                        "status": {"name": "Test Execustion Started"}
                    }
                }
            },
            timeout=30.0,
        )
        duration = time.time() - start_time
        log_api_call(logger, "PATCH", f"{NOTION_API_BASE}/pages/{page_id}", 
                    resp.status_code, duration)
        
        logger.info("   ✓ Marked as 'Test Execustion Started'")
        print("   ✓ Marked as 'Test Execustion Started'")
    except Exception as e:
        log_exception(logger, e, "mark_execution_started")
        logger.warning(f"   ⚠️ Could not set 'Test Execustion Started': {e}")
        print(f"   ⚠️ Could not set 'Test Execustion Started': {e}")

    # Run the bridge
    start_ts = time.monotonic()
    conversation_text, number_of_turns, per_turn_scores, bridge_failed = run_playwright_bridge(
        test_case_name=test_case_name,
        persona=persona,
        test_case_details=test_case_details,
        test_case_prompt=test_case_prompt,
    )
    duration = time.monotonic() - start_ts

    logger.info(f"   ⏱ Execution duration: {duration:.1f} seconds")
    print(f"   ⏱ Execution duration: {duration:.1f} seconds")

    if number_of_turns is not None:
        logger.info(f"   🔄 Number of turns: {number_of_turns}")
        print(f"   🔄 Number of turns: {number_of_turns}")

    if per_turn_scores:
        logger.info(f"   📈 Per-turn scores: {per_turn_scores}")
        print(f"   📈 Per-turn scores: {per_turn_scores}")

    # Append per-turn scores metadata to conversation text for KPI retrieval
    if per_turn_scores:
        import json as _json
        conversation_text += f"\n\n--- Per-turn completeness scores ---\n{_json.dumps(per_turn_scores)}"

    # Create conversation page (always — even on failure it's useful for debugging)
    failed_prefix = "[FAILED] " if bridge_failed else ""
    conv_title = f"{failed_prefix}Conversation – {run_number or ''} – {test_case_name or page_id}"
    logger.info(f"Creating conversation page: {conv_title}")

    conv_page_id = create_formatted_conversation_page(
        parent_page_id=CONVERSATIONS_PARENT_PAGE_ID,
        title=conv_title,
        conversation_text=conversation_text,
    )

    # Decide terminal status: success → EXECUTION_SUCCESS_STATUS ("Test Executed"),
    # failure → EXECUTION_FAILED_STATUS (defaults to keeping it in the "Started"
    # bucket so the evaluator doesn't pick it up and pollute KPIs).
    if bridge_failed:
        terminal_status = EXECUTION_FAILED_STATUS
        logger.error(
            f"   ❌ Bridge reported failure — leaving status as '{terminal_status}' "
            f"(will NOT be evaluated). Conversation page saved for debugging."
        )
        print(
            f"   ❌ Bridge reported failure — status kept as '{terminal_status}'. "
            f"Conversation page saved for debugging."
        )
    else:
        terminal_status = EXECUTION_SUCCESS_STATUS

    # Update execution row
    update_execution_row_with_results(
        execution_page_id=page_id,
        status=terminal_status,
        duration_seconds=duration,
        conversation_page_id=conv_page_id,
        number_of_turns=number_of_turns,
    )

    if bridge_failed:
        logger.error(f"✗ Execution {page_id} failed (see conversation page for details)")
        return False

    logger.info(f"✓ Execution {page_id} completed successfully")
    return True


# =====================================
# CONTINUOUS MONITORING LOOP
# =====================================

def verify_env() -> bool:
    """Verify all required environment variables are set"""
    logger.debug("Verifying environment variables")
    missing = []
    if not NOTION_API_KEY:
        missing.append("NOTION_API_KEY")
    if not TEST_CASE_EXECUTIONS_PAGE_ID:
        missing.append("TEST_CASE_EXECUTIONS_DB_ID (page or db)")
    if not CONVERSATIONS_PARENT_PAGE_ID:
        missing.append("CONVERSATIONS_PARENT_PAGE_ID (Conversations page)")

    if missing:
        logger.error("❌ Missing required environment variables:")
        print("❌ Missing required environment variables:")
        for m in missing:
            logger.error(f"   • {m}")
            print(f"   • {m}")
        return False

    logger.info("✓ All required environment variables present")
    return True


def run_execution_loop(check_interval: int = 10):
    """
    Continuously monitor for pending test executions and dispatch them to
    a pool of EXECUTION_PARALLELISM worker threads.

    Threading model
    ---------------
    The main thread polls Notion every `check_interval` seconds. Each pending
    execution it finds is handed to a ThreadPoolExecutor via submit(). Each
    worker calls process_single_execution(), which is self-contained:
        1. marks the execution as "Test Execustion Started"
        2. spawns the bridge subprocess and waits for it
        3. uploads the conversation page
        4. updates the row to SUCCESS_STATUS or FAILED_STATUS
    Workers spend most of their wall-clock time inside subprocess.run() on
    the bridge, which releases the GIL, so Python threads work well here.

    Race avoidance
    --------------
    Between polling iterations, Notion may still report an execution as
    "Not started" for a few seconds after a worker has picked it up but
    before the "Test Execustion Started" PATCH propagates. We track an
    `in_flight` set of execution IDs locally and skip any execution whose
    ID is already being processed. This plus the server-side Notion PATCH
    gives us at-most-once dispatch.

    Back-pressure
    -------------
    If all EXECUTION_PARALLELISM slots are full, we simply don't submit
    more work that iteration and re-query next tick. No queue buildup —
    each iteration only fills empty slots. Prevents runaway chromium
    spawning when Mila is slow.
    """
    logger.info("=" * 80)
    if EXECUTION_PARALLELISM == 1:
        logger.info("🚀 Test Execution Service Started (Serial Mode, workers=1)")
        print("🚀 Test Execution Service Started (Serial Mode, workers=1)")
    else:
        logger.info(
            f"🚀 Test Execution Service Started (Parallel Mode, workers={EXECUTION_PARALLELISM})"
        )
        logger.warning(
            f"⚠️  PARALLELISM > 1 with a single MILA account is RISKY. Mila's "
            f"xi-deepchat backend may key conversation state by user_id, which "
            f"would cause cross-contamination between concurrent workers. Only "
            f"use this if you have confirmed MILA isolates per session."
        )
        print(
            f"🚀 Test Execution Service Started (Parallel Mode, workers={EXECUTION_PARALLELISM})"
        )
        print(
            f"⚠️  WARNING: parallelism > 1 with a single MILA account is risky — "
            f"MILA may mix conversation state across parallel workers."
        )

    if not verify_env():
        return

    # Auto-discover DB ID from page
    test_exec_db_id = find_database_in_page(TEST_CASE_EXECUTIONS_PAGE_ID)
    logger.info(f"   • Test Case Executions DB: {test_exec_db_id}")
    print(f"   • Test Case Executions DB: {test_exec_db_id}")

    if not test_exec_db_id:
        logger.error("❌ Could not resolve Test Case Executions DB ID")
        print("❌ Could not resolve Test Case Executions DB ID.")
        return

    logger.info(f"⏰ Checking every {check_interval} seconds...")
    print(f"⏰ Checking every {check_interval} seconds...\n")

    iteration = 0
    pool = ThreadPoolExecutor(
        max_workers=EXECUTION_PARALLELISM,
        thread_name_prefix="test-exec",
    )
    in_flight: Dict[str, Future] = {}  # execution_page_id -> Future

    def _worker_wrap(exec_dict: Dict[str, Any], db_id: str, page_id: str) -> bool:
        """Thread target. Wraps process_single_execution with try/except so
        a worker crash can't take down the whole service."""
        try:
            return process_single_execution(exec_dict, db_id)
        except Exception as e:
            log_exception(logger, e, f"worker_process_{page_id}")
            logger.error(f"❌ Worker crashed on execution {page_id}: {e}")
            return False

    try:
        while True:
            try:
                iteration += 1
                logger.debug(f"Check iteration #{iteration}")

                # --- 1. Reap completed workers --------------------------
                for eid, fut in list(in_flight.items()):
                    if fut.done():
                        try:
                            ok = fut.result()
                            if ok:
                                logger.info(f"✅ Worker finished execution {eid}")
                            else:
                                logger.warning(f"⚠️ Worker finished execution {eid} with failure")
                        except Exception as e:
                            log_exception(logger, e, f"reap_{eid}")
                            logger.error(f"❌ Worker result raised: {e}")
                        del in_flight[eid]

                slots_free = EXECUTION_PARALLELISM - len(in_flight)
                logger.debug(
                    f"in_flight={len(in_flight)}/{EXECUTION_PARALLELISM}, "
                    f"slots_free={slots_free}"
                )

                if slots_free <= 0:
                    # All workers busy — don't even bother polling Notion.
                    # This is back-pressure: we won't spawn more chromiums
                    # than the pool's capacity, even if 28 executions are
                    # pending.
                    print(
                        f"🔥 All {EXECUTION_PARALLELISM} workers busy ({', '.join(in_flight.keys())[:80]}…)",
                        end="\r",
                    )
                    time.sleep(check_interval)
                    continue

                # --- 2. Fetch pending executions ------------------------
                pending = get_pending_executions(test_exec_db_id)

                if not pending and not in_flight:
                    logger.debug("💤 No pending executions, no workers active")
                    print("💤 No pending executions...", end="\r")
                    time.sleep(check_interval)
                    continue

                # --- 3. Dispatch up to slots_free new workers -----------
                dispatched = 0
                for exec_dict in pending:
                    if slots_free <= 0:
                        break
                    page_id = exec_dict.get("id", "")
                    if not page_id:
                        continue
                    if page_id in in_flight:
                        # Already being processed — skip (race: Notion may
                        # return it again until the status PATCH propagates).
                        continue

                    logger.info(
                        f"🚀 Dispatching execution {page_id} to worker "
                        f"({len(in_flight)+1}/{EXECUTION_PARALLELISM} slots used)"
                    )
                    print(
                        f"\n🚀 Dispatching {page_id[:8]}… "
                        f"({len(in_flight)+1}/{EXECUTION_PARALLELISM} slots)"
                    )
                    fut = pool.submit(_worker_wrap, exec_dict, test_exec_db_id, page_id)
                    in_flight[page_id] = fut
                    slots_free -= 1
                    dispatched += 1

                if dispatched == 0 and in_flight:
                    # Nothing new was dispatched but we still have workers
                    # running — show a compact progress line.
                    print(
                        f"🔄 {len(in_flight)} worker(s) running, 0 dispatched this tick",
                        end="\r",
                    )
                elif dispatched > 0:
                    logger.info(
                        f"✓ Dispatched {dispatched} new execution(s). "
                        f"in_flight={len(in_flight)}/{EXECUTION_PARALLELISM}"
                    )

                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.warning("\n👋 Shutting down...")
                print("\n\n👋 Shutting down...")
                break
            except Exception as e:
                log_exception(logger, e, "execution loop")
                logger.error(f"\n❌ Error in execution loop: {e}")
                print(f"\n❌ Error in execution loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(check_interval)
    finally:
        logger.info(f"Shutting down thread pool ({len(in_flight)} workers still running)")
        print(f"Shutting down thread pool ({len(in_flight)} workers still running)")
        # wait=True lets each in-flight bridge finish naturally; cancel_futures
        # kills work that hasn't started yet (there shouldn't be any since we
        # never queue beyond EXECUTION_PARALLELISM).
        pool.shutdown(wait=True, cancel_futures=True)


# =====================================
# MAIN
# =====================================

def main():
    logger.info("=" * 80)
    logger.info("MAIN EXECUTION STARTED (Continuous Mode)")
    logger.info("=" * 80)
    
    run_execution_loop(check_interval=CHECK_INTERVAL)


if __name__ == "__main__":
    logger.info("Script started from command line")
    
    try:
        main()
        logger.info("✅ Script completed successfully")
    except KeyboardInterrupt:
        logger.warning("⚠️ Script interrupted by user (Ctrl+C)")
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        log_exception(logger, e, "main script execution")
        logger.error(f"❌ Script failed: {e}")
        raise
    finally:
        logger.info("Script execution finished")