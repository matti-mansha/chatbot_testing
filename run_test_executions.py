# run_test_executions.py
import os
import time
import re
import pathlib
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
TEST_CASE_EXECUTIONS_PAGE_ID = os.getenv("TEST_CASE_EXECUTIONS_DB_ID")  # may be page OR db

# Conversations parent page (from your URL)
# https://www.notion.so/setaro/Conversations-2c7af83cf3e480538ec2c0cef1e96de6
CONVERSATIONS_PARENT_PAGE_ID = os.getenv(
    "CONVERSATIONS_PARENT_PAGE_ID")

# Playwright bridge script
BRIDGE_SCRIPT = os.getenv("BRIDGE_SCRIPT", "playwright_bridge_bot.py")

# Notion API
NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

logger.info("✓ Starting Test Case Executions Runner")
logger.info(f"Bridge Script: {BRIDGE_SCRIPT}")
logger.info(f"Conversations Parent: {CONVERSATIONS_PARENT_PAGE_ID}")
print("✓ Starting Test Case Executions Runner")


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

        # No child database; maybe the given ID is itself a database
        logger.debug("No child database found, using page ID as database ID")
        return page_id_uuid
        
    except Exception as e:
        log_exception(logger, e, "find_database_in_page")
        logger.warning(f"⚠️ Error finding database in page {page_id}: {e}")
        # Fallback: assume it's a database
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
            # ✅ FIX: Concatenate ALL chunks, not just the first one
            lst = prop.get("rich_text", [])
            if not lst:
                return ""
            # Concatenate all rich_text chunks
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
    
    NEW: Added number_of_turns parameter to track conversation turns.
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
    
    # ✅ Add Number of Turns if available
    if number_of_turns is not None:
        properties["Number of Turns"] = {
            "number": number_of_turns
        }
        logger.debug(f"  Added Number of Turns: {number_of_turns}")

    if conversation_page_id:
        url = notion_page_url(conversation_page_id)
        # Conversation flow is rich_text in your schema
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
            print("   Response:", resp.text)  # type: ignore[name-defined]
        except Exception:
            pass


def get_pending_executions(db_id: str) -> List[Dict[str, Any]]:
    """
    Query Test Case Executions DB for rows where:
      - Test Case Status = Active
      - Test Execution Status = Not started
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
            }
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
                print("   Response:", resp.text)  # type: ignore[name-defined]
            except Exception:
                pass
            break

    logger.info(f"✓ Found {len(executions)} pending executions (across {page_count} pages)")
    return executions


# =====================================
# PLAYWRIGHT RUNNER
# =====================================

def extract_turn_count_from_output(output: str) -> Optional[int]:
    """
    Extract number of turns from the bridge script output.
    Looks for patterns like "Number of turns: 5" or "Completed 5 turns"
    """
    logger.debug("Extracting turn count from output")
    
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


def run_playwright_bridge(
    test_case_name: str = "",
    persona: str = "",
    test_case_details: str = "",
    test_case_prompt: str = "",
) -> Tuple[str, Optional[int]]:
    """
    Run playwright_bridge_bot.py as a subprocess and capture stdout as conversation transcript.
    Passes test case parameters as command-line arguments.
    
    Returns: (conversation_text, number_of_turns)
    
    NEW: Extracts number of turns from the script output.
    """
    logger.info(f"Running Playwright bridge script: {BRIDGE_SCRIPT}")
    logger.debug(f"  Test case: {test_case_name}")
    logger.debug(f"  Persona: {persona}")
    logger.debug(f"  Details length: {len(test_case_details)} chars")
    logger.debug(f"  Prompt length: {len(test_case_prompt)} chars")
    
    # Pass parameters as command-line arguments (more direct than env vars)
    cmd = [
        "python", 
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
        
        logger.info(f"✓ Bridge script completed in {duration:.1f}s (exit code: {proc.returncode})")
        
        if proc.returncode != 0:
            logger.error(f"   ❌ Bridge script exited with code {proc.returncode}")
            logger.debug(f"   stderr:\n{proc.stderr}")
            print(f"   ❌ Bridge script exited with code {proc.returncode}")
            print("   stderr:\n", proc.stderr)
        else:
            print("   ✓ Bridge script completed")

        # Combine stdout + stderr as conversation trace (or keep only stdout)
        conversation_text = proc.stdout.strip()
        if not conversation_text:
            conversation_text = f"(No stdout captured. stderr was:)\n\n{proc.stderr}"
            logger.warning("No stdout captured from bridge script")

        logger.debug(f"Captured conversation: {len(conversation_text)} chars")
        
        # ✅ Extract turn count from output
        number_of_turns = extract_turn_count_from_output(conversation_text)
        if number_of_turns is not None:
            logger.info(f"✓ Extracted turn count: {number_of_turns}")
            print(f"   ✓ Extracted turn count: {number_of_turns}")
        else:
            logger.warning("⚠️ Could not extract turn count from output")
            print("   ⚠️ Could not extract turn count from output")
        
        return conversation_text, number_of_turns
        
    except Exception as e:
        log_exception(logger, e, "run_playwright_bridge")
        logger.error(f"   ❌ Error running bridge script: {e}")
        print(f"   ❌ Error running bridge script: {e}")
        return f"(Bridge error: {e})", None


# =====================================
# MAIN LOOP
# =====================================

def verify_env() -> bool:
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


def main():
    logger.info("=" * 80)
    logger.info("MAIN EXECUTION STARTED")
    logger.info("=" * 80)
    
    if not verify_env():
        logger.error("Environment verification failed, exiting")
        return

    # Auto-discover DB ID from page (same style as prepare_test_runs.py)
    test_exec_db_id = find_database_in_page(TEST_CASE_EXECUTIONS_PAGE_ID)
    logger.info(f"   • Test Case Executions DB: {test_exec_db_id}")
    print(f"   • Test Case Executions DB: {test_exec_db_id}")

    if not test_exec_db_id:
        logger.error("❌ Could not resolve Test Case Executions DB ID")
        print("❌ Could not resolve Test Case Executions DB ID.")
        return

    # Fetch pending executions
    pending = get_pending_executions(test_exec_db_id)

    if not pending:
        logger.info("No pending executions. Exiting.")
        print("No pending executions. Exiting.")
        return

    logger.info(f"🔔 Found {len(pending)} pending execution(s)")
    print(f"🔔 Found {len(pending)} pending execution(s).")

    for idx, execution in enumerate(pending, start=1):
        page_id = execution.get("id")
        logger.info("\n" + "=" * 60)
        logger.info(f"▶ [{idx}/{len(pending)}] Processing execution {page_id}")
        logger.info("=" * 60)
        print("\n" + "=" * 60)
        print(f"▶ [{idx}/{len(pending)}] Processing execution {page_id}")
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

        # Mark as "Test Execustion Started" BEFORE running bridge (if such status exists)
        try:
            start_time = time.time()
            resp = httpx.patch(
                f"{NOTION_API_BASE}/pages/{page_id}",
                headers=HEADERS,
                json={
                    "properties": {
                        "Test Execution Status": {
                            # NOTE: option name has typo in your schema: 'Test Execustion Started'
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

        start_ts = time.monotonic()
        conversation_text, number_of_turns = run_playwright_bridge(
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

        # Create conversation page under Conversations parent
        # NOW USING THE IMPROVED FORMATTER!
        conv_title = f"Conversation – {run_number or ''} – {test_case_name or page_id}"
        logger.info(f"Creating conversation page: {conv_title}")
        
        conv_page_id = create_formatted_conversation_page(
            parent_page_id=CONVERSATIONS_PARENT_PAGE_ID,
            title=conv_title,
            conversation_text=conversation_text,
        )

        # Update execution row with turn count
        update_execution_row_with_results(
            execution_page_id=page_id,
            status="Test Executed",
            duration_seconds=duration,
            conversation_page_id=conv_page_id,
            number_of_turns=number_of_turns,  # ✅ Pass turn count
        )

    logger.info("\n" + "=" * 80)
    logger.info("✅ All pending executions processed")
    logger.info("=" * 80)
    print("\n✅ All pending executions processed.")


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