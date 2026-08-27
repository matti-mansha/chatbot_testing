# prepare_test_runs.py
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx

from logging_config import setup_logging, log_exception, log_api_call

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging("prepare_test_runs")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
TEST_CASES_PAGE_ID = os.getenv("TEST_CASES_DB_ID")  # Actually page IDs
TEST_RUNS_PAGE_ID = os.getenv("TEST_RUNS_DB_ID")
TEST_CASE_EXECUTIONS_PAGE_ID = os.getenv("TEST_CASE_EXECUTIONS_DB_ID")

# Prompt parent pages
TEST_CASE_PROMPT_PARENT_PAGE_ID = os.getenv("TEST_CASE_PROMPT_PARENT_PAGE_ID")
EVALUATION_PROMPT_PARENT_PAGE_ID = os.getenv("EVALUATION_PROMPT_PARENT_PAGE_ID")

# Notion API base URL and headers
NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

logger.info("✓ Starting Test Run Preparation Service")
print(f"✓ Starting Test Run Preparation Service")


def format_notion_id(notion_id: str) -> str:
    """Convert to UUID format with hyphens."""
    clean_id = notion_id.replace("-", "")
    if len(clean_id) == 32:
        return f"{clean_id[0:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:32]}"
    return notion_id


def find_database_in_page(page_id: str) -> Optional[str]:
    """
    Given a page ID, find the database block inside it.
    Returns the database ID.
    """
    logger.debug(f"Finding database in page: {page_id}")
    page_id = format_notion_id(page_id)
    
    try:
        # Get blocks from the page
        start_time = time.time()
        response = httpx.get(
            f"{NOTION_API_BASE}/blocks/{page_id}/children",
            headers=HEADERS,
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "GET", f"{NOTION_API_BASE}/blocks/{page_id}/children", 
                    response.status_code, duration)
        
        response.raise_for_status()
        blocks = response.json()
        
        for block in blocks.get("results", []):
            block_type = block.get("type")
            
            # Check for database blocks
            if block_type in ["child_database", "table"]:
                db_id = block["id"]
                logger.info(f"Found {block_type} in page: {db_id}")
                return db_id
        
        # If no database found in blocks, the page itself might be a database
        logger.debug("No child database found, using page ID as database ID")
        return page_id
        
    except Exception as e:
        log_exception(logger, e, "find_database_in_page")
        logger.warning(f"⚠️ Error finding database in page {page_id}: {e}")
        # Fallback: assume page ID is database ID
        return page_id


def get_child_pages(parent_page_id: str) -> List[Dict[str, Any]]:
    """Returns child_page blocks under a parent page."""
    logger.debug(f"Getting child pages for: {parent_page_id}")
    parent_page_id = format_notion_id(parent_page_id)
    
    all_blocks: List[Dict[str, Any]] = []
    cursor = None
    page_count = 0

    while True:
        params = {}
        if cursor:
            params["start_cursor"] = cursor
        
        start_time = time.time()
        response = httpx.get(
            f"{NOTION_API_BASE}/blocks/{parent_page_id}/children",
            headers=HEADERS,
            params=params,
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "GET", f"{NOTION_API_BASE}/blocks/{parent_page_id}/children", 
                    response.status_code, duration)
        
        response.raise_for_status()
        resp = response.json()
        
        page_blocks = resp.get("results", [])
        all_blocks.extend(page_blocks)
        page_count += 1
        
        logger.debug(f"Page {page_count}: Found {len(page_blocks)} blocks")
        
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    child_pages = [b for b in all_blocks if b.get("type") == "child_page"]
    logger.info(f"Found {len(child_pages)} child pages (from {len(all_blocks)} total blocks)")
    return child_pages


def get_latest_child_page_id(parent_page_id: str) -> Optional[str]:
    """Among child pages, find the one with the most recent last_edited_time."""
    logger.debug(f"Finding latest child page for: {parent_page_id}")
    child_pages = get_child_pages(parent_page_id)

    if not child_pages:
        logger.warning("No child pages found")
        return None

    latest_id = None
    latest_time = None

    for block in child_pages:
        page_id = block["id"]
        
        # Retrieve page to get last_edited_time
        start_time = time.time()
        response = httpx.get(
            f"{NOTION_API_BASE}/pages/{page_id}",
            headers=HEADERS,
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "GET", f"{NOTION_API_BASE}/pages/{page_id}", 
                    response.status_code, duration)
        
        response.raise_for_status()
        page = response.json()
        
        last_edited = page.get("last_edited_time")
        if last_edited is None:
            continue
        if latest_time is None or last_edited > latest_time:
            latest_time = last_edited
            latest_id = page_id

    if latest_id:
        logger.info(f"Latest child page: {latest_id} (edited: {latest_time})")
    
    return latest_id


def split_text_for_notion(text: str, max_length: int = 1900) -> List[Dict[str, Any]]:
    """
    Split long text into chunks for Notion rich_text (max 2000 chars per block).
    
    FIXED: Changed max_length from 2000 to 1900 to provide a safety buffer and 
    avoid edge cases that could result in chunks slightly over 2000 characters.
    
    Returns a list of rich_text objects.
    """
    if not text:
        return [{"text": {"content": ""}}]
    
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        chunks.append({
            "text": {"content": chunk}
        })
    
    logger.debug(f"Split text into {len(chunks)} chunks (max {max_length} chars each)")
    return chunks


def load_text_from_page_blocks(page_id: str) -> str:
    """
    Concatenate text from paragraphs & headings of a given page.
    """
    logger.debug(f"Loading text from page: {page_id}")
    page_id = format_notion_id(page_id)
    
    lines: List[str] = []
    cursor = None
    page_count = 0

    while True:
        params = {}
        if cursor:
            params["start_cursor"] = cursor
        
        start_time = time.time()
        response = httpx.get(
            f"{NOTION_API_BASE}/blocks/{page_id}/children",
            headers=HEADERS,
            params=params,
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "GET", f"{NOTION_API_BASE}/blocks/{page_id}/children", 
                    response.status_code, duration)
        
        response.raise_for_status()
        resp = response.json()
        
        page_count += 1
        
        for block in resp.get("results", []):
            btype = block.get("type")
            data = block.get(btype, {})

            if btype in ("paragraph", "heading_1", "heading_2", "heading_3"):
                rich_text = data.get("rich_text", [])
                txt = "".join(rt.get("plain_text", "") for rt in rich_text)
                if txt.strip():
                    lines.append(txt)

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    result = "\n\n".join(lines).strip()
    logger.info(f"Loaded {len(result)} chars from page (across {page_count} pages)")
    return result


def load_prompt_from_notion_parent_page(parent_page_id: str) -> str:
    """
    Gets the latest child page from a parent page and returns its text content.
    """
    if not parent_page_id:
        return ""
    
    try:
        logger.debug(f"Loading prompt from parent page: {parent_page_id}")
        parent_page_id = format_notion_id(parent_page_id)
        latest_page_id = get_latest_child_page_id(parent_page_id)
        
        if not latest_page_id:
            logger.warning(f"⚠️ No child pages found under {parent_page_id}")
            print(f"⚠️ No child pages found under {parent_page_id}")
            return ""
        
        content = load_text_from_page_blocks(latest_page_id)
        logger.info(f"Loaded prompt: {len(content)} chars")
        return content
        
    except Exception as e:
        log_exception(logger, e, "load_prompt_from_notion_parent_page")
        logger.warning(f"⚠️ Error loading prompt from {parent_page_id}: {e}")
        print(f"⚠️ Error loading prompt from {parent_page_id}: {e}")
        return ""


# Auto-discover actual database IDs from page IDs
logger.info("🔍 Auto-discovering database IDs from page IDs...")
print("🔍 Auto-discovering database IDs from page IDs...")

TEST_CASES_DB_ID = find_database_in_page(TEST_CASES_PAGE_ID)
TEST_RUNS_DB_ID = find_database_in_page(TEST_RUNS_PAGE_ID)
TEST_CASE_EXECUTIONS_DB_ID = find_database_in_page(TEST_CASE_EXECUTIONS_PAGE_ID)

logger.info(f"   ✓ Test Cases DB: {TEST_CASES_DB_ID}")
logger.info(f"   ✓ Test Runs DB: {TEST_RUNS_DB_ID}")
logger.info(f"   ✓ Test Executions DB: {TEST_CASE_EXECUTIONS_DB_ID}")
print(f"   ✓ Test Cases DB: {TEST_CASES_DB_ID}")
print(f"   ✓ Test Runs DB: {TEST_RUNS_DB_ID}")
print(f"   ✓ Test Executions DB: {TEST_CASE_EXECUTIONS_DB_ID}")

# Load prompts from Notion at startup
logger.info("📄 Loading prompts from Notion parent pages...")
print("📄 Loading prompts from Notion parent pages...")

TEST_CASE_PROMPT_CONTENT = load_prompt_from_notion_parent_page(TEST_CASE_PROMPT_PARENT_PAGE_ID)
EVALUATION_PROMPT_CONTENT = load_prompt_from_notion_parent_page(EVALUATION_PROMPT_PARENT_PAGE_ID)

if TEST_CASE_PROMPT_CONTENT:
    logger.info(f"   ✓ Test Case Prompt loaded ({len(TEST_CASE_PROMPT_CONTENT)} chars)")
    print(f"   ✓ Test Case Prompt loaded ({len(TEST_CASE_PROMPT_CONTENT)} chars)")
else:
    logger.warning(f"   ⚠️ Test Case Prompt is empty")
    print(f"   ⚠️ Test Case Prompt is empty")

if EVALUATION_PROMPT_CONTENT:
    logger.info(f"   ✓ Evaluation Prompt loaded ({len(EVALUATION_PROMPT_CONTENT)} chars)")
    print(f"   ✓ Evaluation Prompt loaded ({len(EVALUATION_PROMPT_CONTENT)} chars)")
else:
    logger.warning(f"   ⚠️ Evaluation Prompt is empty")
    print(f"   ⚠️ Evaluation Prompt is empty")


def get_test_runs_to_prepare() -> List[Dict[str, Any]]:
    """Query Test Runs table for entries with status = "Prepare Test Run"""
    logger.debug("Querying for test runs to prepare")
    
    try:
        start_time = time.time()
        response = httpx.post(
            f"{NOTION_API_BASE}/databases/{TEST_RUNS_DB_ID}/query",
            headers=HEADERS,
            json={
                "filter": {
                    "property": "Status",
                    "status": {
                        "equals": "Prepare Test Run"
                    }
                }
            },
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "POST", f"{NOTION_API_BASE}/databases/{TEST_RUNS_DB_ID}/query", 
                    response.status_code, duration)
        
        response.raise_for_status()
        results = response.json().get("results", [])
        logger.info(f"Found {len(results)} test runs to prepare")
        return results
        
    except Exception as e:
        log_exception(logger, e, "get_test_runs_to_prepare")
        logger.error(f"❌ Error querying Test Runs: {e}")
        print(f"❌ Error querying Test Runs: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_all_test_cases() -> List[Dict[str, Any]]:
    """Query Test cases DB for ALL entries (both Active and Inactive) with pagination"""
    logger.info("Fetching all test cases (Active and Inactive)")
    all_results = []
    cursor = None
    page_num = 0
    
    try:
        while True:
            page_num += 1
            body = {
                "filter": {
                    "property": "Status",
                    "status": {"equals": "Active"}
                }
            }
            if cursor:
                body["start_cursor"] = cursor
            
            start_time = time.time()
            response = httpx.post(
                f"{NOTION_API_BASE}/databases/{TEST_CASES_DB_ID}/query",
                headers=HEADERS,
                json=body,
                timeout=30.0
            )
            duration = time.time() - start_time
            log_api_call(logger, "POST", f"{NOTION_API_BASE}/databases/{TEST_CASES_DB_ID}/query", 
                        response.status_code, duration)
            
            response.raise_for_status()
            data = response.json()
            
            # Add results from this page
            results = data.get("results", [])
            all_results.extend(results)
            
            logger.debug(f"   📄 Page {page_num}: Fetched {len(results)} test cases (total: {len(all_results)})")
            print(f"   📄 Fetched {len(all_results)} test cases so far...")
            
            # Check if there are more pages
            has_more = data.get("has_more", False)
            if not has_more:
                break
            
            # Get cursor for next page
            cursor = data.get("next_cursor")
        
        logger.info(f"✓ Fetched {len(all_results)} total test cases across {page_num} pages")
        return all_results
        
    except Exception as e:
        log_exception(logger, e, "get_all_test_cases")
        logger.error(f"❌ Error querying Test Cases: {e}")
        print(f"❌ Error querying Test Cases: {e}")
        import traceback
        traceback.print_exc()
        return all_results  # Return what we got so far


def extract_property_value(page: Dict[str, Any], property_name: str) -> Any:
    """Extract value from a Notion property based on its type"""
    try:
        prop = page["properties"].get(property_name, {})
        prop_type = prop.get("type")
        
        if prop_type == "title":
            title_list = prop.get("title", [])
            return title_list[0]["plain_text"] if title_list else ""
        elif prop_type == "rich_text":
            # ✅ FIX: Concatenate ALL chunks, not just the first one
            rich_text_list = prop.get("rich_text", [])
            if not rich_text_list:
                return ""
            # Concatenate all rich_text chunks
            full_text = "".join(item.get("plain_text", "") for item in rich_text_list)
            return full_text
        elif prop_type == "number":
            return prop.get("number")
        elif prop_type == "select":
            select = prop.get("select")
            return select["name"] if select else None
        elif prop_type == "status":
            status = prop.get("status")
            return status["name"] if status else None
        elif prop_type == "relation":
            return prop.get("relation", [])
        elif prop_type == "date":
            date = prop.get("date")
            return date["start"] if date else None
        else:
            return None
    except Exception as e:
        logger.warning(f"⚠️ Error extracting property '{property_name}': {e}")
        print(f"⚠️ Error extracting property '{property_name}': {e}")
        return None


def create_test_case_execution(
    test_run_page_id: str,
    test_run_number: int,
    test_case: Dict[str, Any],
    general_prompt: str
) -> Optional[str]:
    """Create a Test Case Execution entry, preserving the original test case status"""
    try:
        # Extract test case details
        test_case_name = extract_property_value(test_case, "Test case name")
        test_case_id = extract_property_value(test_case, "Test case ID")
        persona = extract_property_value(test_case, "Persona")
        test_case_details = extract_property_value(test_case, "Test Case Details")
        evaluation_criteria = extract_property_value(test_case, "Evaluation Criteria")
        test_case_status = extract_property_value(test_case, "Status")
        
        logger.debug(f"Creating execution for: {test_case_name} (ID: {test_case_id}, Status: {test_case_status})")
        
        # Use the loaded prompts from Notion parent pages
        test_case_prompt = TEST_CASE_PROMPT_CONTENT
        evaluation_prompt = EVALUATION_PROMPT_CONTENT
        
        # Create the execution entry with CORRECT property types
        properties = {
            "Test run number": {
                "title": [
                    {
                        "text": {
                            "content": f"{test_run_number}.{test_case_id}" if test_run_number and test_case_id else "Unknown"
                        }
                    }
                ]
            },
            "Test case name": {
                "rich_text": split_text_for_notion(test_case_name or f"Test Case {test_case_id}")
            },
            "Persona": {
                "rich_text": split_text_for_notion(persona or "")
            },
            "Test Case Details": {
                "rich_text": split_text_for_notion(test_case_details or "")
            },
            "Evaluation Criteria": {
                "rich_text": split_text_for_notion(evaluation_criteria or "")
            },
            "Test Case Prompt": {
                "rich_text": split_text_for_notion(test_case_prompt)
            },
            "Test Case Evaluation Prompt": {
                "rich_text": split_text_for_notion(evaluation_prompt)
            },
            "Execution date / time": {
                "date": {
                    "start": datetime.now().isoformat()
                }
            }
        }
        
        # Add Test Case Status if it exists (Active or In Active)
        if test_case_status:
            # Map "Inactive" to "In Active" if needed
            status_value = test_case_status
            if status_value == "Inactive":
                status_value = "In Active"
            
            properties["Test Case Status"] = {
                "status": {
                    "name": status_value
                }
            }

        # Test Execution Status must be set EXPLICITLY — do not rely on Notion's
        # column default.
        #
        # run_test_executions.py filters for `Test Execution Status = "Not started"`.
        # A duplicated or migrated workspace can lose the column's default value, in
        # which case rows are created with a null status and the executor's filter can
        # never match them. The pipeline then polls forever, logs
        # "Found 0 pending executions", raises no error, and silently does nothing.
        # Observed on the AuPairWorld handover workspace, 2026-08-24.
        properties["Test Execution Status"] = {
            "status": {
                "name": "Not started"
            }
        }

        # Create page using HTTP POST
        start_time = time.time()
        response = httpx.post(
            f"{NOTION_API_BASE}/pages",
            headers=HEADERS,
            json={
                "parent": {"database_id": TEST_CASE_EXECUTIONS_DB_ID},
                "properties": properties
            },
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "POST", f"{NOTION_API_BASE}/pages", response.status_code, duration)
        
        response.raise_for_status()
        new_page = response.json()
        
        status_label = f"[{test_case_status}]" if test_case_status else ""
        logger.info(f"   ✓ Created execution for: {test_case_name} {status_label}")
        print(f"   ✓ Created execution for: {test_case_name} {status_label}")
        return new_page["id"]
        
    except Exception as e:
        test_case_name = extract_property_value(test_case, "Test case name") or "Unknown"
        log_exception(logger, e, f"create_test_case_execution for {test_case_name}")
        logger.error(f"   ❌ Error creating execution for {test_case_name}: {e}")
        print(f"   ❌ Error creating execution for {test_case_name}: {e}")
        
        # Print response body for debugging
        if hasattr(e, 'response'):
            try:
                error_body = e.response.json()
                logger.debug(f"   Error details: {error_body}")
                print(f"   Error details: {error_body}")
            except:
                pass
        
        import traceback
        traceback.print_exc()
        return None


def update_test_run_status(test_run_page_id: str, status: str, total_test_cases: int = 0):
    """Update Test Run status and metadata"""
    logger.info(f"Updating test run {test_run_page_id} to status: {status}")
    
    try:
        properties = {
            "Status": {
                "status": {
                    "name": status
                }
            }
        }
        
        if total_test_cases > 0:
            properties["Total number of test cases"] = {
                "number": total_test_cases
            }
        
        start_time = time.time()
        response = httpx.patch(
            f"{NOTION_API_BASE}/pages/{test_run_page_id}",
            headers=HEADERS,
            json={"properties": properties},
            timeout=30.0
        )
        duration = time.time() - start_time
        log_api_call(logger, "PATCH", f"{NOTION_API_BASE}/pages/{test_run_page_id}", 
                    response.status_code, duration)
        
        response.raise_for_status()
        
        logger.info(f"✓ Updated Test Run status to: {status} (total cases: {total_test_cases})")
        print(f"✓ Updated Test Run status to: {status}")
        
    except Exception as e:
        log_exception(logger, e, "update_test_run_status")
        logger.error(f"❌ Error updating Test Run status: {e}")
        print(f"❌ Error updating Test Run status: {e}")
        import traceback
        traceback.print_exc()


def prepare_test_run(test_run: Dict[str, Any]):
    """Main function to prepare a test run by copying ALL test cases (Active and Inactive)"""
    test_run_page_id = test_run["id"]
    test_run_number = extract_property_value(test_run, "Test Run Number")
    general_prompt = extract_property_value(test_run, "General conversation prompt used in this run")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 Preparing Test Run #{test_run_number}")
    logger.info(f"{'='*60}")
    print(f"\n{'='*60}")
    print(f"📋 Preparing Test Run #{test_run_number}")
    print(f"{'='*60}")
    
    # Get ALL test cases
    logger.info("🔍 Fetching all test cases...")
    print("🔍 Fetching all test cases...")
    all_test_cases = get_all_test_cases()
    
    if not all_test_cases:
        logger.warning("⚠️ No test cases found!")
        print("⚠️ No test cases found!")
        update_test_run_status(test_run_page_id, "Start Test Run", 0)
        return
    
    # Count by status
    active_count = sum(1 for tc in all_test_cases if extract_property_value(tc, "Status") == "Active")
    inactive_count = sum(1 for tc in all_test_cases if extract_property_value(tc, "Status") == "Inactive")
    
    logger.info(f"✓ Found {len(all_test_cases)} test cases total:")
    logger.info(f"  • {active_count} Active")
    logger.info(f"  • {inactive_count} Inactive")
    print(f"✓ Found {len(all_test_cases)} test cases total:")
    print(f"  • {active_count} Active")
    print(f"  • {inactive_count} Inactive")
    
    # Create executions for each test case
    logger.info(f"\n📝 Creating test case executions...")
    print(f"\n📝 Creating test case executions...")
    created_count = 0
    skipped_count = 0
    
    for i, test_case in enumerate(all_test_cases, 1):
        test_case_name = extract_property_value(test_case, "Test case name")
        test_case_id = extract_property_value(test_case, "Test case ID")
        test_case_status = extract_property_value(test_case, "Status")
        
        # Skip if Test case ID is empty
        if not test_case_id:
            logger.warning(f"   [{i}/{len(all_test_cases)}] {test_case_name} - ⚠️ SKIPPED (empty Test case ID)")
            print(f"   [{i}/{len(all_test_cases)}] {test_case_name} - ⚠️ SKIPPED (empty Test case ID)")
            skipped_count += 1
            continue
        
        logger.debug(f"   [{i}/{len(all_test_cases)}] {test_case_name} [{test_case_status}]")
        print(f"   [{i}/{len(all_test_cases)}] {test_case_name} [{test_case_status}]")
        
        execution_id = create_test_case_execution(
            test_run_page_id=test_run_page_id,
            test_run_number=test_run_number,
            test_case=test_case,
            general_prompt=general_prompt or ""
        )
        
        if execution_id:
            created_count += 1
        
        time.sleep(0.3)
    
    # Update test run status to "Start Test Run"
    logger.info(f"\n✅ Created {created_count}/{len(all_test_cases)} test case executions")
    logger.info(f"  • {active_count} Active test cases")
    logger.info(f"  • {inactive_count} Inactive test cases")
    print(f"\n✅ Created {created_count}/{len(all_test_cases)} test case executions")
    print(f"  • {active_count} Active test cases")
    print(f"  • {inactive_count} Inactive test cases")
    
    if skipped_count > 0:
        logger.warning(f"  • {skipped_count} skipped (empty Test case ID)")
        print(f"  • {skipped_count} skipped (empty Test case ID)")
    
    update_test_run_status(test_run_page_id, "Start Test Run", created_count)
    logger.info(f"{'='*60}\n")
    print(f"{'='*60}\n")


def verify_environment():
    """Verify all required environment variables are set"""
    logger.debug("Verifying environment variables")
    
    required_vars = {
        "NOTION_API_KEY": NOTION_API_KEY,
        "TEST_CASES_DB_ID (page)": TEST_CASES_PAGE_ID,
        "TEST_RUNS_DB_ID (page)": TEST_RUNS_PAGE_ID,
        "TEST_CASE_EXECUTIONS_DB_ID (page)": TEST_CASE_EXECUTIONS_PAGE_ID
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    
    if missing:
        logger.error("❌ Missing required environment variables:")
        print("❌ Missing required environment variables:")
        for var in missing:
            logger.error(f"   • {var}")
            print(f"   • {var}")
        return False
    
    logger.info("✓ All environment variables set")
    print("✓ All environment variables set")
    return True


def run_preparation_loop(check_interval: int = 10):
    """Continuously monitor for test runs that need preparation"""
    logger.info("=" * 80)
    logger.info("🚀 Test Run Preparation Service Started")
    logger.info("=" * 80)
    print("🚀 Test Run Preparation Service Started")
    
    if not verify_environment():
        return
    
    logger.info(f"⏰ Checking every {check_interval} seconds...")
    print(f"⏰ Checking every {check_interval} seconds...\n")
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            logger.debug(f"Check iteration #{iteration}")
            
            test_runs_to_prepare = get_test_runs_to_prepare()
            
            if test_runs_to_prepare:
                logger.info(f"🔔 Found {len(test_runs_to_prepare)} test run(s) to prepare")
                print(f"🔔 Found {len(test_runs_to_prepare)} test run(s) to prepare")
                for test_run in test_runs_to_prepare:
                    prepare_test_run(test_run)
            else:
                # Only log to file, print to console with \r to overwrite
                logger.debug("💤 No test runs to prepare")
                print("💤 No test runs to prepare...", end="\r")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.warning("\n👋 Shutting down...")
            print("\n\n👋 Shutting down...")
            break
        except Exception as e:
            log_exception(logger, e, "preparation loop")
            logger.error(f"\n❌ Error: {e}")
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(check_interval)


if __name__ == "__main__":
    logger.info("Script started from command line")
    
    try:
        run_preparation_loop(check_interval=10)
        logger.info("✅ Script completed successfully")
    except KeyboardInterrupt:
        logger.warning("⚠️ Script interrupted by user (Ctrl+C)")
    except Exception as e:
        log_exception(logger, e, "main script execution")
        logger.error(f"❌ Script failed: {e}")
        raise
    finally:
        logger.info("Script execution finished")