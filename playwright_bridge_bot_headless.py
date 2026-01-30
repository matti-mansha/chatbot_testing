# playwright_bridge_bot_headless.py
"""
Headless version of the Playwright bridge.
Uses HTTP API instead of Streamlit for the tester bot.

ENHANCEMENTS:
- Extensive debug logging for every operation
- Detailed browser state tracking
- Step-by-step message exchange logging
- Network activity tracking
- Screenshot capture on errors (optional)
- Full diagnostic capture system
"""
import os
import time
import pathlib
import sys
import httpx
from typing import Optional, List, Dict, Tuple
from diagnostic_utils import DiagnosticCapture
from playwright.sync_api import (
    sync_playwright,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)
from dotenv import load_dotenv
from logging_config import setup_logging, log_exception, log_api_call

# =====================================
# LOAD .env
# =====================================

BASE_DIR = pathlib.Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# Set up logging
logger = setup_logging("playwright_bridge")

MILA_URL = os.getenv("MILA_URL", "").strip()
TESTER_API_URL = os.getenv("TESTER_API_URL", "http://localhost:8501")
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

# ✅ NEW: Enhanced debugging options
ENABLE_SCREENSHOTS = os.getenv("ENABLE_SCREENSHOTS", "false").lower() == "true"
SCREENSHOT_DIR = BASE_DIR / "screenshots"
DETAILED_TIMING = os.getenv("DETAILED_TIMING", "true").lower() == "true"

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
TESTER_API_TIMEOUT = float(os.getenv("TESTER_API_TIMEOUT", "120"))

# Mila web login credentials
MILA_LOGIN_USER = os.getenv("MILA_LOGIN_USER", "").strip()
MILA_LOGIN_PASS = os.getenv("MILA_LOGIN_PASS", "").strip()

# Optional HTTP basic auth
MILA_HTTP_USER = os.getenv("MILA_HTTP_USER")
MILA_HTTP_PASS = os.getenv("MILA_HTTP_PASS")

# Test case parameters from command line
TEST_CASE = sys.argv[1] if len(sys.argv) > 1 else "Onboarding & \"How it works\""
TEST_PERSONA = sys.argv[2] if len(sys.argv) > 2 else "Host family in Spain"
TEST_CASE_DETAILS = sys.argv[3] if len(sys.argv) > 3 else ""
TEST_CASE_PROMPT = sys.argv[4] if len(sys.argv) > 4 else ""

# Create screenshot directory if needed
if ENABLE_SCREENSHOTS:
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    logger.info(f"📸 Screenshots enabled: {SCREENSHOT_DIR}")

logger.info("=" * 80)
logger.info("CONFIGURATION LOADED")
logger.info("=" * 80)
logger.info(f"  MILA_URL: {MILA_URL}")
logger.info(f"  TESTER_API_URL: {TESTER_API_URL}")
logger.info(f"  MAX_TURNS: {MAX_TURNS}")
logger.info(f"  MAX_RETRIES: {MAX_RETRIES}")
logger.info(f"  RETRY_DELAY: {RETRY_DELAY}s")
logger.info(f"  TESTER_API_TIMEOUT: {TESTER_API_TIMEOUT}s")
logger.info(f"  Test Case: {TEST_CASE}")
logger.info(f"  Persona: {TEST_PERSONA}")
logger.info(f"  Screenshots: {ENABLE_SCREENSHOTS}")
logger.info(f"  Detailed Timing: {DETAILED_TIMING}")
logger.debug(f"  Test Case Details: {len(TEST_CASE_DETAILS)} chars")
logger.debug(f"  Test Case Prompt: {len(TEST_CASE_PROMPT)} chars")
logger.info("=" * 80)

# Conversation log
CONVERSATION_LOG: List[Dict[str, str]] = []

# Session ID for the test bot API
tester_session_id: Optional[str] = None

# Screenshot counter
screenshot_counter = 0


# =====================================
# UTILITY FUNCTIONS
# =====================================

def take_screenshot(page: Page, label: str):
    """Take a screenshot for debugging"""
    if not ENABLE_SCREENSHOTS:
        return
    
    global screenshot_counter
    screenshot_counter += 1
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{screenshot_counter:03d}_{timestamp}_{label}.png"
    filepath = SCREENSHOT_DIR / filename
    
    try:
        page.screenshot(path=str(filepath))
        logger.info(f"📸 Screenshot saved: {filename}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save screenshot: {e}")


def log_timing(operation: str, start_time: float):
    """Log operation timing if detailed timing is enabled"""
    if DETAILED_TIMING:
        duration = time.time() - start_time
        logger.debug(f"⏱️  {operation}: {duration:.3f}s")


def log_page_state(page: Page, label: str):
    """Log current page state for debugging"""
    try:
        url = page.url
        title = page.title()
        logger.debug(f"📄 Page State ({label}):")
        logger.debug(f"   URL: {url}")
        logger.debug(f"   Title: {title}")
        
        # Log visible elements count
        try:
            bubble_count = len(page.query_selector_all(".message-bubble"))
            logger.debug(f"   Message bubbles: {bubble_count}")
        except:
            pass
            
    except Exception as e:
        logger.warning(f"⚠️ Could not log page state: {e}")


# =====================================
# SELECTORS (Mila only)
# =====================================

SELECTORS = {
    "mila": {
        "message_bubbles": ".message-bubble.ai-message.ai-message-text",
        "input_selectors": [
            "input[placeholder='Enter a prompt here']",
            "textarea[placeholder='Enter a prompt here']",
            "[contenteditable='true']",
            "div.ai-deepchat--input-container input",
            "div.ai-deepchat--input-container textarea",
            "textarea.ai-deepchat--textarea",
            "input[type='text']:not([name='q'])",
            "div[class*='deepchat'] input",
            "div[class*='deepchat'] textarea",
        ],
        "send_button": (
            "button.ai-deepchat--send, "
            "button[aria-label='Send'], "
            "button:has-text('Send')"
        ),
        "header": "div.ai-deepchat--header",
        "clear_history": "a.clear-history, a[class*='clear-history'], a.chat-dropdown-link",
    }
}

logger.debug("Selectors configured:")
for key, value in SELECTORS["mila"].items():
    if isinstance(value, list):
        logger.debug(f"  {key}: {len(value)} options")
    else:
        logger.debug(f"  {key}: {value[:50]}...")


# =====================================
# TESTER BOT HTTP API FUNCTIONS
# =====================================

def check_tester_health() -> bool:
    """Check if tester API is healthy"""
    logger.info("=" * 60)
    logger.info("CHECKING TESTER API HEALTH")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        response = httpx.get(
            f"{TESTER_API_URL}/health",
            timeout=10.0
        )
        duration = time.time() - start_time
        
        log_api_call(logger, "GET", f"{TESTER_API_URL}/health", 
                    response.status_code, duration)
        
        response.raise_for_status()
        data = response.json()
        
        logger.info("✓ Health check passed")
        logger.info(f"  Status: {data.get('status')}")
        logger.info(f"  Active sessions: {data.get('active_sessions')}")
        logger.info(f"  OpenAI available: {data.get('openai_available')}")
        logger.info(f"  OpenAI model: {data.get('openai_model')}")
        logger.info("=" * 60)
        
        print(f"✓ Tester API health check: OK")
        print(f"  Active sessions: {data.get('active_sessions')}")
        
        return data.get('status') == 'ok'
        
    except Exception as e:
        log_exception(logger, e, "check_tester_health")
        logger.error("=" * 60)
        logger.error(f"❌ Health check failed: {e}")
        logger.error("=" * 60)
        print(f"❌ Tester API health check failed: {e}")
        return False


def create_tester_session() -> Optional[str]:
    """Create a new test session via HTTP API"""
    global tester_session_id
    
    logger.info("=" * 60)
    logger.info("CREATING TESTER SESSION")
    logger.info("=" * 60)
    
    # Check health first
    if not check_tester_health():
        logger.error("❌ Cannot create session - API unhealthy")
        print("❌ Tester API is not healthy")
        return None
    
    logger.info("Requesting new session from API")
    logger.info(f"  Test case: {TEST_CASE}")
    logger.info(f"  Persona: {TEST_PERSONA}")
    logger.debug(f"  Details length: {len(TEST_CASE_DETAILS)} chars")
    logger.debug(f"  Prompt length: {len(TEST_CASE_PROMPT)} chars")
    
    try:
        start_time = time.time()
        response = httpx.post(
            f"{TESTER_API_URL}/session/create",
            json={
                "test_case": TEST_CASE,
                "persona": TEST_PERSONA,
                "test_case_details": TEST_CASE_DETAILS,
                "test_case_prompt": TEST_CASE_PROMPT
            },
            timeout=30.0
        )
        duration = time.time() - start_time
        
        log_api_call(logger, "POST", f"{TESTER_API_URL}/session/create", 
                    response.status_code, duration)
        
        response.raise_for_status()
        data = response.json()
        tester_session_id = data.get("session_id")
        
        logger.info("=" * 60)
        logger.info(f"✓ Session created: {tester_session_id}")
        logger.info(f"  API response time: {duration:.2f}s")
        logger.info("=" * 60)
        
        print(f"✓ Created tester session: {tester_session_id}")
        return tester_session_id
        
    except Exception as e:
        log_exception(logger, e, "create_tester_session")
        logger.error("=" * 60)
        logger.error(f"❌ Session creation failed: {e}")
        logger.error("=" * 60)
        print(f"❌ Error creating tester session: {e}")
        return None


def send_to_tester_api(message: str, retry_count: int = 0) -> Optional[Tuple[str, Optional[int], bool]]:
    """
    Send message to tester bot via HTTP API with retry logic.
    
    Returns: (reply, score, should_continue) or None on error
    """
    if not tester_session_id:
        logger.error("❌ No active tester session")
        print("❌ No active tester session")
        return None
    
    logger.info("=" * 60)
    logger.info(f"SENDING MESSAGE TO TESTER API (Attempt {retry_count + 1}/{MAX_RETRIES})")
    logger.info("=" * 60)
    logger.info(f"Session: {tester_session_id}")
    logger.info(f"Message length: {len(message)} chars")
    logger.debug(f"Message preview: {message[:200]}...")
    
    try:
        start_time = time.time()
        response = httpx.post(
            f"{TESTER_API_URL}/session/{tester_session_id}/message",
            json={"message": message},
            timeout=TESTER_API_TIMEOUT,
        )
        duration = time.time() - start_time
        
        log_api_call(logger, "POST", 
                    f"{TESTER_API_URL}/session/{tester_session_id}/message", 
                    response.status_code, duration)
        
        # Handle different status codes
        if response.status_code == 404:
            logger.error("❌ Session not found on server")
            logger.error("   Session may have been cleaned up or expired")
            print("❌ Session not found on server")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        reply = data.get("reply", "")
        score = data.get("score")
        should_continue = data.get("should_continue", True)
        error_count = data.get("error_count", 0)
        turn = data.get("turn", "?")
        
        logger.info("✓ Received reply from API")
        logger.info(f"  Turn: {turn}")
        logger.info(f"  Reply length: {len(reply)} chars")
        logger.info(f"  Score: {score}")
        logger.info(f"  Should continue: {should_continue}")
        logger.info(f"  Error count: {error_count}")
        logger.info(f"  API response time: {duration:.2f}s")
        logger.debug(f"  Reply preview: {reply[:200]}...")
        logger.info("=" * 60)
        
        # Check if reply indicates an error
        if reply.startswith("❌"):
            logger.warning(f"⚠️ Tester returned error response")
            logger.warning(f"   Error: {reply}")
            
            # If it's a timeout or rate limit, retry
            if "timeout" in reply.lower() or "rate limit" in reply.lower():
                if retry_count < MAX_RETRIES - 1:
                    logger.info(f"⏳ Retrying after {RETRY_DELAY}s...")
                    print(f"⏳ Retrying after {RETRY_DELAY}s (attempt {retry_count + 2}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                    return send_to_tester_api(message, retry_count + 1)
        
        return reply, score, should_continue
        
    except httpx.TimeoutException as e:
        log_exception(logger, e, "send_to_tester_api (timeout)")
        logger.error("=" * 60)
        logger.error(f"❌ Timeout after {TESTER_API_TIMEOUT}s")
        logger.error("=" * 60)
        print(f"❌ Timeout after {TESTER_API_TIMEOUT}s")
        
        # Retry on timeout
        if retry_count < MAX_RETRIES - 1:
            logger.info(f"⏳ Retrying after {RETRY_DELAY}s...")
            print(f"⏳ Retrying after {RETRY_DELAY}s (attempt {retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            return send_to_tester_api(message, retry_count + 1)
        return None
        
    except httpx.HTTPStatusError as e:
        log_exception(logger, e, "send_to_tester_api (HTTP error)")
        logger.error("=" * 60)
        logger.error(f"❌ HTTP error {e.response.status_code}")
        logger.error(f"   Response: {e.response.text[:500]}")
        logger.error("=" * 60)
        print(f"❌ HTTP error {e.response.status_code}: {e}")
        
        # Retry on 5xx errors (server errors)
        if 500 <= e.response.status_code < 600 and retry_count < MAX_RETRIES - 1:
            logger.info(f"⏳ Retrying after {RETRY_DELAY}s...")
            print(f"⏳ Retrying after {RETRY_DELAY}s (attempt {retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            return send_to_tester_api(message, retry_count + 1)
        return None
        
    except Exception as e:
        log_exception(logger, e, "send_to_tester_api")
        logger.error("=" * 60)
        logger.error(f"❌ Unexpected error: {type(e).__name__}")
        logger.error(f"   {e}")
        logger.error("=" * 60)
        print(f"❌ Error sending to tester API: {e}")
        
        # Generic retry
        if retry_count < MAX_RETRIES - 1:
            logger.info(f"⏳ Retrying after {RETRY_DELAY}s...")
            print(f"⏳ Retrying after {RETRY_DELAY}s (attempt {retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            return send_to_tester_api(message, retry_count + 1)
        return None


def cleanup_tester_session():
    """Delete the tester session when done"""
    if not tester_session_id:
        return
    
    logger.info("=" * 60)
    logger.info(f"CLEANING UP SESSION: {tester_session_id}")
    logger.info("=" * 60)
    print(f"🧹 Cleaning up tester session...")
    
    try:
        start_time = time.time()
        response = httpx.delete(
            f"{TESTER_API_URL}/session/{tester_session_id}",
            timeout=10.0
        )
        duration = time.time() - start_time
        
        log_api_call(logger, "DELETE", 
                    f"{TESTER_API_URL}/session/{tester_session_id}", 
                    response.status_code, duration)
        
        response.raise_for_status()
        
        logger.info("✓ Session cleaned up successfully")
        logger.info(f"  Response time: {duration:.2f}s")
        logger.info("=" * 60)
        print(f"✓ Cleaned up tester session")
        
    except Exception as e:
        log_exception(logger, e, "cleanup_tester_session")
        logger.warning("=" * 60)
        logger.warning(f"⚠️ Could not clean up session: {e}")
        logger.warning("=" * 60)
        print(f"⚠️ Could not clean up session: {e}")


# =====================================
# MILA HELPERS
# =====================================

def wait_for_mila_typing_complete(page: Page, selector: str, timeout_ms: int = 60000):
    """Wait for Mila to finish typing"""
    logger.info("=" * 60)
    logger.info("WAITING FOR MILA TO FINISH TYPING")
    logger.info("=" * 60)
    logger.debug(f"Selector: {selector}")
    logger.debug(f"Timeout: {timeout_ms}ms")
    
    print("⏳ Waiting for Mila to finish typing...")
    
    start_time = time.time()
    end_time = time.time() + (timeout_ms / 1000)
    last_text = None
    stable_count = 0
    check_count = 0
    
    while time.time() < end_time:
        check_count += 1
        elapsed = time.time() - start_time
        
        try:
            elems = page.query_selector_all(selector)
            logger.debug(f"Check #{check_count} ({elapsed:.1f}s): Found {len(elems)} elements")
            
            if elems:
                last_elem = elems[-1]
                current_text = last_elem.inner_text().strip()
                
                logger.debug(f"  Current text length: {len(current_text)} chars")
                logger.debug(f"  Text preview: {current_text[:50]}...")
                
                if current_text in ("...", "• • •", ""):
                    logger.debug("  Still typing (dots/empty)")
                    last_text = None
                    stable_count = 0
                    time.sleep(0.5)
                    continue
                
                if current_text == last_text:
                    stable_count += 1
                    logger.debug(f"  Text stable (count: {stable_count}/2)")
                    if stable_count >= 2:
                        logger.info("✓ Text stabilized - typing complete")
                        logger.info(f"  Total wait time: {elapsed:.2f}s")
                        logger.info(f"  Checks performed: {check_count}")
                        logger.info("=" * 60)
                        print("✓ Mila finished typing")
                        return
                else:
                    stable_count = 0
                    logger.debug("  Text changed - still typing")
                    last_text = current_text
                
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Check #{check_count} failed: {e}")
            time.sleep(0.5)
    
    logger.warning("⚠️ Typing wait timeout reached")
    logger.warning(f"  Total wait time: {time.time() - start_time:.2f}s")
    logger.warning(f"  Checks performed: {check_count}")
    logger.warning("=" * 60)
    print("⚠️ Typing wait timeout, proceeding anyway...")

def handle_mila_technical_error(page: Page, max_wait_ms: int = 2000) -> bool:
    """
    Check for technical errors and automatically click retry button.
    Returns True if retry button was clicked, False otherwise.
    """
    logger.debug("Checking for technical error state...")
    
    try:
        # Wait a moment for error to appear
        page.wait_for_timeout(max_wait_ms)
        
        # Look for error message
        error_found = False
        error_selectors = [
            "text='I am sorry, something went terribly wrong'",
            "text='Please try to ask me again'",
        ]
        
        for selector in error_selectors:
            try:
                if page.locator(selector).first.is_visible(timeout=1000):
                    error_found = True
                    logger.info(f"✓ Detected technical error message")
                    print(f"⚠️ Technical error detected in Mila response")
                    break
            except:
                continue
        
        if not error_found:
            logger.debug("  No error message detected")
            return False
        
        # Look for retry button
        retry_selectors = [
            "button:has-text('Retry last instruction')",
            "button:has-text('Retry')",
            "button:has-text('Try again')",
        ]
        
        for selector in retry_selectors:
            try:
                retry_btn = page.locator(selector).first
                if retry_btn.is_visible(timeout=2000):
                    logger.info(f"🔄 Found retry button: {selector}")
                    print(f"🔄 Clicking Retry button...")
                    retry_btn.click()
                    page.wait_for_timeout(1500)
                    logger.info("✓ Clicked retry button")
                    return True
            except Exception as e:
                logger.debug(f"  Retry selector failed: {e}")
                continue
        
        logger.warning("⚠️ Error detected but no retry button found")
        return False
        
    except Exception as e:
        logger.debug(f"Error check failed: {e}")
        return False


def get_mila_last_message_text(page: Page, selector: str, retry_count: int = 3) -> Optional[str]:
    """Get the last message text from Mila"""
    logger.info("=" * 60)
    logger.info("EXTRACTING MILA'S LAST MESSAGE")
    logger.info("=" * 60)
    logger.debug(f"Selector: {selector}")
    logger.debug(f"Max retries: {retry_count}")
    
    for attempt in range(retry_count):
        logger.debug(f"Attempt {attempt + 1}/{retry_count}")
        
        try:
            elems = page.query_selector_all(selector)
            logger.debug(f"  Found {len(elems)} message elements")
            
            if not elems:
                if attempt < retry_count - 1:
                    logger.warning(f"  No elements found, retrying...")
                    print(f"   ⚠️ No message elements found, retry {attempt + 1}/{retry_count}")
                    time.sleep(1)
                    continue
                logger.error("  No elements found after all retries")
                return None
            
            last_elem = elems[-1]
            time.sleep(0.5)
            
            # Try inner_text
            try:
                text = last_elem.inner_text().strip()
                logger.debug(f"  inner_text: {len(text)} chars")
                if text and text not in ["...", "• • •"]:
                    logger.info("✓ Successfully extracted message via inner_text")
                    logger.info(f"  Length: {len(text)} chars")
                    logger.debug(f"  Preview: {text[:100]}...")
                    logger.info("=" * 60)
                    return text
                else:
                    logger.debug(f"  inner_text returned invalid text: '{text}'")
            except Exception as e:
                logger.debug(f"  inner_text failed: {e}")
            
            # Try text_content
            try:
                text = last_elem.text_content().strip()
                logger.debug(f"  text_content: {len(text)} chars")
                if text and text not in ["...", "• • •"]:
                    logger.info("✓ Successfully extracted message via text_content")
                    logger.info(f"  Length: {len(text)} chars")
                    logger.debug(f"  Preview: {text[:100]}...")
                    logger.info("=" * 60)
                    return text
                else:
                    logger.debug(f"  text_content returned invalid text: '{text}'")
            except Exception as e:
                logger.debug(f"  text_content failed: {e}")
            
            # Try evaluate
            try:
                text = last_elem.evaluate("el => el.innerText || el.textContent").strip()
                logger.debug(f"  evaluate: {len(text)} chars")
                if text and text not in ["...", "• • •"]:
                    logger.info("✓ Successfully extracted message via evaluate")
                    logger.info(f"  Length: {len(text)} chars")
                    logger.debug(f"  Preview: {text[:100]}...")
                    logger.info("=" * 60)
                    return text
                else:
                    logger.debug(f"  evaluate returned invalid text: '{text}'")
            except Exception as e:
                logger.debug(f"  evaluate failed: {e}")
            
            if attempt < retry_count - 1:
                logger.warning(f"  All extraction methods failed, retrying...")
                print(f"   ⚠️ All extraction methods failed, retry {attempt + 1}/{retry_count}")
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"  Extraction attempt {attempt + 1} failed: {e}")
            log_exception(logger, e, f"get_mila_last_message_text attempt {attempt + 1}")
            if attempt < retry_count - 1:
                time.sleep(1)
    
    logger.error("❌ Failed to extract message after all retries")
    logger.error("=" * 60)
    return None


def wait_for_new_message(page: Page, selector: str, previous_count: int, timeout_ms=30000):
    """
    Wait for a NEW message bubble to appear (bubble count must INCREASE).
    """
    logger.info("=" * 60)
    logger.info("WAITING FOR NEW MESSAGE BUBBLE")
    logger.info("=" * 60)
    logger.info(f"Previous count: {previous_count}")
    logger.info(f"Timeout: {timeout_ms}ms")
    logger.debug(f"Selector: {selector}")
    
    print(f"⏳ Waiting for new message bubble (current: {previous_count})...")
    
    start_time = time.time()
    end_time = time.time() + (timeout_ms / 1000)
    last_logged_count = previous_count
    check_count = 0
    
    # Wait a moment for UI to settle
    time.sleep(0.5)
    
    while time.time() < end_time:
        check_count += 1
        elapsed = time.time() - start_time
        current_count = len(page.query_selector_all(selector))
        
        # Log count changes
        if current_count != last_logged_count:
            logger.debug(f"Check #{check_count} ({elapsed:.1f}s): Count changed {last_logged_count} → {current_count}")
            last_logged_count = current_count
        elif check_count % 10 == 0:  # Log every 10th check
            logger.debug(f"Check #{check_count} ({elapsed:.1f}s): Still waiting (count: {current_count})")
        
        # New bubble appeared!
        if current_count > previous_count:
            logger.info("✓ New message bubble detected!")
            logger.info(f"  Previous: {previous_count}")
            logger.info(f"  Current: {current_count}")
            logger.info(f"  Wait time: {elapsed:.2f}s")
            logger.info(f"  Checks: {check_count}")
            logger.info("=" * 60)
            print(f"✓ New message bubble appeared! (was {previous_count}, now {current_count})")
            return current_count
        
        time.sleep(0.3)
    
    # Timeout - no new bubble appeared
    final_count = len(page.query_selector_all(selector))
    logger.error("❌ TIMEOUT: No new message bubble appeared")
    logger.error(f"  Expected: > {previous_count}")
    logger.error(f"  Got: {final_count}")
    logger.error(f"  Wait time: {time.time() - start_time:.2f}s")
    logger.error(f"  Checks: {check_count}")
    logger.error("=" * 60)
    
    raise PlaywrightTimeoutError(f"No new message bubble appeared in {timeout_ms}ms (count stayed at {final_count})")


def dismiss_cookie_banner(page: Page):
    """Dismiss cookie banners"""
    logger.info("=" * 60)
    logger.info("DISMISSING COOKIE BANNER")
    logger.info("=" * 60)
    
    for label in ["Accept all", "Reject non-essential services", "Customise settings"]:
        try:
            logger.debug(f"Looking for button: '{label}'")
            btn = page.get_by_role("button", name=label)
            if btn.is_visible():
                logger.info(f"✓ Found and clicking: '{label}'")
                print(f"➡️ Clicking cookie banner: {label}")
                btn.click()
                page.wait_for_timeout(800)
                logger.info("✓ Cookie banner dismissed")
                logger.info("=" * 60)
                return
        except Exception as e:
            logger.debug(f"  Button '{label}' not found or not clickable: {e}")

    try:
        logger.debug("Looking for generic Accept button")
        btn = page.locator("button", has_text="Accept")
        if btn.first.is_visible():
            logger.info("✓ Found and clicking generic Accept button")
            print("➡️ Clicking generic Accept cookie button")
            btn.first.click()
            page.wait_for_timeout(800)
            logger.info("✓ Cookie banner dismissed")
            logger.info("=" * 60)
            return
    except Exception as e:
        logger.debug(f"  Generic Accept button not found: {e}")

    logger.info("ℹ️ No cookie banner found or already dismissed")
    logger.info("=" * 60)
    print("ℹ️ Cookie banner not found or already dismissed.")


def perform_mila_login(page: Page):
    """Perform Mila login"""
    logger.info("=" * 60)
    logger.info("PERFORMING MILA LOGIN")
    logger.info("=" * 60)
    
    if not MILA_LOGIN_USER or not MILA_LOGIN_PASS:
        logger.warning("⚠️ Credentials not set - skipping login")
        logger.info("=" * 60)
        print("⚠️ MILA_LOGIN_USER / MILA_LOGIN_PASS not set — skipping login.")
        return

    try:
        logger.info("🔐 Waiting for login form...")
        print("🔐 Waiting for login form...")

        # Find username field
        logger.debug("Looking for username field")
        username_locator = page.get_by_label(
            "Login by username/email address", exact=False
        )
        if username_locator.count() == 0:
            logger.debug("  Label not found, trying generic input")
            username_locator = page.locator(
                "input:not([type='password']):not([type='hidden']):not([type='checkbox']):not([type='radio'])"
            ).first
        username_locator.wait_for(timeout=8000)
        logger.info("✓ Username field found")

        # Find password field
        logger.debug("Looking for password field")
        password_locator = page.get_by_label("Password", exact=False)
        if password_locator.count() == 0:
            logger.debug("  Label not found, trying input[type='password']")
            password_locator = page.locator("input[type='password']").first
        password_locator.wait_for(timeout=8000)
        logger.info("✓ Password field found")

        # Fill username
        logger.info(f"➡️ Filling username: {MILA_LOGIN_USER}")
        print(f"➡️ Filling username: {MILA_LOGIN_USER}")
        username_locator.click()
        username_locator.fill(MILA_LOGIN_USER)
        logger.debug("  Username filled")

        # Fill password
        logger.info("➡️ Filling password")
        print("➡️ Filling password")
        password_locator.click()
        password_locator.fill(MILA_LOGIN_PASS)
        logger.debug("  Password filled")

        # Click login
        logger.info("➡️ Clicking Log in button")
        print("➡️ Clicking Log in button...")
        login_btn = page.get_by_role("button", name="Log in")
        login_btn.click()

        page.wait_for_timeout(2000)
        logger.info("✓ Login attempt complete")
        logger.info("=" * 60)
        print("🔓 Login attempt complete.")
        
    except PlaywrightTimeoutError:
        logger.info("ℹ️ Login form did not appear (maybe already logged in)")
        logger.info("=" * 60)
        print("ℹ️ Login form did not appear (maybe already logged in?).")
    except Exception as e:
        log_exception(logger, e, "perform_mila_login")
        logger.error("=" * 60)
        print(f"❌ Error during login: {e}")


def open_mila_chat(page: Page):
    """Open Mila chat widget"""
    logger.info("=" * 60)
    logger.info("OPENING MILA CHAT WIDGET")
    logger.info("=" * 60)
    
    try:
        logger.debug(f"Looking for header: {SELECTORS['mila']['header']}")
        header = page.locator(SELECTORS["mila"]["header"]).first
        header.wait_for(state="visible", timeout=10000)
        logger.debug("  Header found")
        
        aria_expanded = header.get_attribute("aria-expanded")
        logger.debug(f"  aria-expanded: {aria_expanded}")
        
        if aria_expanded == "false":
            logger.info("💬 Chat widget closed, opening...")
            print("💬 Opening Mila chat widget...")
            header.click()
            page.wait_for_timeout(800)
            logger.info("✓ Chat widget opened")
        else:
            logger.info("💬 Chat widget already open")
            print("💬 Mila chat widget already open.")
        
        logger.info("=" * 60)
        take_screenshot(page, "chat_opened")
        
    except Exception as e:
        log_exception(logger, e, "open_mila_chat")
        logger.warning("=" * 60)
        print(f"ℹ️ Could not open Mila chat widget automatically: {e}")


def clear_mila_history(page: Page):
    """Clear Mila chat history"""
    logger.info("=" * 60)
    logger.info("CLEARING MILA CHAT HISTORY")
    logger.info("=" * 60)
    
    try:
        print("🗑️  Clearing Mila chat history...")
        
        # Open menu
        menu_opened = False
        menu_selectors = [
            "svg.chevron-icon",
            "svg#chevron",
            "button[aria-label*='menu']",
            "button[aria-label*='Menu']",
            ".ai-deepchat--header svg",
            ".ai-deepchat--header button",
            "[class*='menu-icon']",
            "[class*='dropdown']",
        ]
        
        logger.debug("Looking for menu button")
        for menu_sel in menu_selectors:
            try:
                logger.debug(f"  Trying selector: {menu_sel}")
                menu_btn = page.locator(menu_sel).first
                if menu_btn.is_visible(timeout=2000):
                    logger.info(f"✓ Found menu button: {menu_sel}")
                    print(f"   Found menu button with selector: {menu_sel}")
                    menu_btn.click()
                    page.wait_for_timeout(800)
                    menu_opened = True
                    logger.debug("  Menu opened")
                    break
            except Exception as e:
                logger.debug(f"  Selector failed: {e}")
                continue
        
        if not menu_opened:
            logger.warning("⚠️ Could not find menu button, trying direct access")
            print("   ⚠️ Could not find menu button, trying direct access...")
        
        # Find Clear History button
        clear_selectors = [
            "a.clear-history",
            "a[class*='clear-history']",
            "a.chat-dropdown-link:has-text('Clear History')",
            "a:has-text('Clear History')",
            "button:has-text('Clear History')",
        ]
        
        clicked = False
        logger.debug("Looking for Clear History button")
        for clear_sel in clear_selectors:
            try:
                logger.debug(f"  Trying selector: {clear_sel}")
                clear_btn = page.locator(clear_sel).first
                if clear_btn.is_visible(timeout=2000):
                    logger.info(f"✓ Found Clear History: {clear_sel}")
                    print(f"   Found Clear History with selector: {clear_sel}")
                    clear_btn.click()
                    logger.info("✓ Clicked 'Clear History'")
                    print("   ✓ Clicked 'Clear History'")
                    page.wait_for_timeout(1500)
                    clicked = True
                    logger.debug("  Waiting for confirmation dialog")
                    break
            except Exception as e:
                logger.debug(f"  Selector failed: {e}")
                continue
        
        if not clicked:
            try:
                logger.debug("Trying text match for Clear History")
                clear_link = page.get_by_text("Clear History", exact=False)
                if clear_link.is_visible(timeout=2000):
                    clear_link.click()
                    logger.info("✓ Clicked via text match")
                    print("   ✓ Clicked 'Clear History' using text match")
                    page.wait_for_timeout(1500)
                    clicked = True
            except Exception as e:
                logger.debug(f"  Text match failed: {e}")
        
        if not clicked:
            logger.error("❌ Could not find 'Clear History' button")
            print("   ⚠️ Could not find 'Clear History' button")
            logger.info("=" * 60)
            return
        
        # Confirm clearing
        try:
            logger.debug("Looking for confirmation button")
            confirm_selectors = [
                "button:has-text('Confirm')",
                "button:has-text('Yes')",
                "button:has-text('OK')",
                "button:has-text('Clear')",
            ]
            
            for confirm_sel in confirm_selectors:
                try:
                    logger.debug(f"  Trying: {confirm_sel}")
                    confirm_btn = page.locator(confirm_sel).first
                    if confirm_btn.is_visible(timeout=2000):
                        confirm_btn.click()
                        logger.info("✓ Confirmed clearing")
                        print("   ✓ Confirmed history clearing")
                        page.wait_for_timeout(1000)
                        break
                except Exception as e:
                    logger.debug(f"  Failed: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Confirmation handling: {e}")
        
        logger.info("✓ History cleared successfully")
        print("   ✓ History cleared successfully")
        logger.info("=" * 60)
        take_screenshot(page, "history_cleared")
        
    except Exception as e:
        log_exception(logger, e, "clear_mila_history")
        logger.error("=" * 60)
        print(f"   ❌ Error clearing Mila history: {e}")


def ensure_mila_chat_open(page: Page):
    """Ensure Mila chat is open"""
    logger.debug("Ensuring Mila chat is open")
    
    try:
        header = page.locator(SELECTORS["mila"]["header"]).first
        if header.is_visible():
            aria_expanded = header.get_attribute("aria-expanded")
            if aria_expanded == "false":
                logger.info("💬 Reopening chat widget")
                print("💬 Reopening Mila chat widget...")
                header.click()
                page.wait_for_timeout(800)
        
        # Scroll to bottom
        page.evaluate("""
            const chatContainer = document.querySelector('div.ai-deepchat--body, div[class*="chat-messages"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        """)
        page.wait_for_timeout(500)
        logger.debug("  Chat scrolled to bottom")
    except Exception as e:
        logger.warning(f"Could not ensure chat is open: {e}")


def find_mila_input(page: Page) -> Optional[any]:
    """Find Mila input field"""
    logger.debug("Finding Mila input field")
    
    for selector in SELECTORS["mila"]["input_selectors"]:
        try:
            logger.debug(f"  Trying: {selector}")
            locator = page.locator(selector).first
            if locator.count() > 0:
                try:
                    locator.wait_for(state="attached", timeout=2000)
                    logger.info(f"✓ Found input: {selector}")
                    print(f"✓ Found Mila input with selector: {selector}")
                    return locator
                except Exception as e:
                    logger.debug(f"    Attached check failed: {e}")
                    continue
        except Exception as e:
            logger.debug(f"    Failed: {e}")
            continue
    
    logger.error("❌ Could not find any input field")
    return None


def send_message_to_mila(page: Page, text: str, max_retries: int = 3):
    """Send message to Mila"""
    logger.info("=" * 60)
    logger.info("SENDING MESSAGE TO MILA")
    logger.info("=" * 60)
    logger.info(f"Message length: {len(text)} chars")
    logger.debug(f"Message preview: {text[:200]}...")
    
    send_sel = SELECTORS["mila"]["send_button"]

    for attempt in range(max_retries):
        logger.debug(f"Send attempt {attempt + 1}/{max_retries}")
        
        try:
            start_time = time.time()
            ensure_mila_chat_open(page)
            
            # Find input
            box = find_mila_input(page)
            if box is None:
                raise Exception("Could not find Mila input field")
            
            box.wait_for(state="visible", timeout=30000)
            logger.debug("  Input field visible")
            box.click()
            page.wait_for_timeout(300)
            
            # Fill input
            try:
                logger.debug("  Filling via JavaScript")
                box.evaluate(f"""
                    (element) => {{
                        if (element.contentEditable === 'true') {{
                            element.textContent = {repr(text)};
                            element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }} else {{
                            element.value = {repr(text)};
                            element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }}
                """)
                page.wait_for_timeout(500)
                logger.debug("  JavaScript fill successful")
            except Exception as e:
                logger.warning(f"  JavaScript fill failed: {e}, trying keyboard")
                box.fill("")
                page.keyboard.type(text, delay=0)
                logger.debug("  Keyboard fill successful")

            # Click send button
            logger.debug("  Looking for send button")
            btn = page.locator(send_sel).first
            try:
                btn.wait_for(state="visible", timeout=5000)
                logger.debug("  Send button visible, clicking")
                btn.click()
            except PlaywrightTimeoutError:
                logger.warning("  Send button not visible, pressing Enter")
                try:
                    page.keyboard.press("Enter")
                except Exception as e:
                    logger.error(f"  Failed to press Enter: {e}")
            
            elapsed = time.time() - start_time
            logger.info("✓ Message sent to Mila")
            logger.info(f"  Send operation took: {elapsed:.2f}s")
            logger.info("=" * 60)
            print("✓ Message sent to Mila")
            take_screenshot(page, "message_sent")
            return
            
        except PlaywrightTimeoutError as e:
            if attempt == max_retries - 1:
                log_exception(logger, e, f"send_message_to_mila final attempt")
                logger.error("=" * 60)
                print(f"❌ Failed after {max_retries} attempts")
                raise
            logger.warning(f"  Attempt {attempt + 1} timed out, retrying...")
            print(f"⚠️ Retry {attempt + 1}/{max_retries}...")
            page.wait_for_timeout(5000)
        except Exception as e:
            if attempt == max_retries - 1:
                log_exception(logger, e, f"send_message_to_mila final attempt")
                logger.error("=" * 60)
                raise
            logger.warning(f"  Attempt {attempt + 1} failed: {e}")
            print(f"⚠️ Retry {attempt + 1}/{max_retries} - Error: {e}")
            page.wait_for_timeout(5000)


# =====================================
# MAIN BRIDGE FUNCTION
# =====================================

def run_bridge() -> Tuple[List[Dict[str, str]], int]:
    """
    Run the headless bridge with extensive logging and diagnostics.
    Returns: (conversation_log, number_of_turns)
    """
    global CONVERSATION_LOG
    CONVERSATION_LOG = []
    
    turns_completed = 0

    # ✨ Initialize diagnostics
    diagnostics = DiagnosticCapture(output_dir="diagnostics")
    logger.info("✓ Diagnostics system initialized")

    logger.info("=" * 80)
    logger.info("STARTING HEADLESS BRIDGE BOT")
    logger.info("=" * 80)
    logger.info(f"Mila URL:           {MILA_URL}")
    logger.info(f"Tester API URL:     {TESTER_API_URL}")
    logger.info(f"Max turns:          {MAX_TURNS}")
    logger.info(f"Max retries:        {MAX_RETRIES}")
    logger.info(f"Retry delay:        {RETRY_DELAY}s")
    logger.info(f"Screenshots:        {ENABLE_SCREENSHOTS}")
    logger.info(f"Detailed timing:    {DETAILED_TIMING}")
    logger.info("=" * 80)
    
    print("\n" + "=" * 80)
    print("Starting headless bridge bot...")
    print(f"Mila URL:        {MILA_URL}")
    print(f"Tester API URL:  {TESTER_API_URL}")
    print(f"Max turns:       {MAX_TURNS}")
    print(f"Max retries:     {MAX_RETRIES}")
    print("=" * 80 + "\n")

    if not MILA_URL:
        logger.error("❌ Missing MILA_URL in .env")
        print("❌ Missing MILA_URL in .env")
        return CONVERSATION_LOG, turns_completed

    # Create tester session
    if not create_tester_session():
        logger.error("❌ Failed to create tester session")
        print("❌ Failed to create tester session")
        return CONVERSATION_LOG, turns_completed

    try:
        logger.info("=" * 60)
        logger.info("LAUNCHING BROWSER")
        logger.info("=" * 60)
        
        with sync_playwright() as p:
            browser_start = time.time()
            logger.debug("Launching Chromium in headless mode")
            logger.debug("  Args: --no-sandbox, --disable-dev-shm-usage")
            
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            browser_launch_time = time.time() - browser_start
            logger.info(f"✓ Browser launched in {browser_launch_time:.2f}s")
            logger.info("=" * 60)

            # Create context
            ctx_kwargs_mila = {
                "viewport": {"width": 1280, "height": 720},
                "ignore_https_errors": True,
            }
            if MILA_HTTP_USER and MILA_HTTP_PASS:
                logger.debug("Adding HTTP credentials to context")
                ctx_kwargs_mila["http_credentials"] = {
                    "username": MILA_HTTP_USER,
                    "password": MILA_HTTP_PASS,
                }

            logger.debug("Creating browser context")
            context_mila = browser.new_context(**ctx_kwargs_mila)
            page_mila = context_mila.new_page()

            # ✨ Setup diagnostic listeners
            diagnostics.setup_listeners(page_mila)
            logger.info("✓ Diagnostic listeners activated")
            
            logger.info("✓ Browser context created")

            # Navigate to Mila
            logger.info("=" * 60)
            logger.info(f"NAVIGATING TO: {MILA_URL}")
            logger.info("=" * 60)
            
            nav_start = time.time()
            page_mila.goto(MILA_URL)
            page_mila.wait_for_load_state("networkidle")
            nav_time = time.time() - nav_start
            
            logger.info(f"✓ Page loaded in {nav_time:.2f}s")
            log_page_state(page_mila, "after_navigation")
            logger.info("=" * 60)
            
            # ✨ Capture initial state
            diagnostics.capture_dom_snapshot(page_mila, "01_page_loaded")
            take_screenshot(page_mila, "page_loaded")

            # Setup steps
            dismiss_cookie_banner(page_mila)
            perform_mila_login(page_mila)
            page_mila.wait_for_timeout(2000)
            open_mila_chat(page_mila)
            
            # ✨ Capture after opening chat
            diagnostics.capture_dom_snapshot(page_mila, "02_chat_opened")
            diagnostics.detect_all_chat_elements(page_mila, "after_chat_open")
            
            clear_mila_history(page_mila)
            
            # ✨ Capture after clearing history
            diagnostics.capture_dom_snapshot(page_mila, "03_history_cleared")

            mila_sel = SELECTORS["mila"]["message_bubbles"]

            # ===== FIRST MILA MESSAGE =====
            logger.info("=" * 60)
            logger.info("WAITING FOR MILA'S FIRST MESSAGE")
            logger.info("=" * 60)
            
            initial_mila_count = len(page_mila.query_selector_all(mila_sel))
            logger.info(f"Initial bubble count: {initial_mila_count}")
            print(f"⏳ Waiting for Mila's first message... (bubbles: {initial_mila_count})")
            
            if initial_mila_count == 0:
                try:
                    wait_for_new_message(page_mila, mila_sel, previous_count=0, timeout_ms=30000)
                except PlaywrightTimeoutError:
                    logger.error("❌ Mila didn't send first message (timeout)")
                    print("❌ Mila didn't send first message")
                    diagnostics.capture_dom_snapshot(page_mila, "04_first_message_timeout_ERROR")
                    take_screenshot(page_mila, "first_message_timeout")
                    browser.close()
                    return CONVERSATION_LOG, turns_completed
            
            try:
                wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=30000)
            except Exception as e:
                logger.warning(f"⚠️ Error waiting for typing: {e}")

            mila_count = len(page_mila.query_selector_all(mila_sel))
            mila_last = get_mila_last_message_text(page_mila, mila_sel)
            
            if not mila_last:
                logger.error("❌ Failed to extract Mila's first message")
                print("❌ Failed to extract Mila's first message")
                diagnostics.capture_dom_snapshot(page_mila, "05_first_message_extraction_ERROR")
                take_screenshot(page_mila, "first_message_extraction_failed")
                browser.close()
                return CONVERSATION_LOG, turns_completed
            
            logger.info("✓ RECEIVED MILA'S FIRST MESSAGE")
            logger.info(f"  Length: {len(mila_last)} chars")
            logger.info(f"  Content: {mila_last[:200]}...")
            print(f"🟠 Mila first message:\n{mila_last}\n")
            CONVERSATION_LOG.append({"speaker": "Mila", "message": mila_last})
            
            # ✨ Capture first message state
            diagnostics.capture_dom_snapshot(page_mila, "06_first_message_received")
            diagnostics.detect_all_chat_elements(page_mila, "first_message")
            take_screenshot(page_mila, "first_message_received")

            # ===== MAIN LOOP =====
            logger.info("=" * 80)
            logger.info(f"STARTING MAIN CONVERSATION LOOP (MAX {MAX_TURNS} TURNS)")
            logger.info("=" * 80)
            
            early_exit = False
            early_exit_reason = ""
            
            for turn in range(1, MAX_TURNS + 1):
                logger.info("\n" + "=" * 80)
                logger.info(f"TURN {turn}/{MAX_TURNS}")
                logger.info("=" * 80)
                print(f"\n{'='*80}\nTURN {turn}/{MAX_TURNS}\n{'='*80}")

                # Mila → Tester
                result = send_to_tester_api(mila_last)
                if result is None:
                    logger.error("❌ Tester API failed, stopping")
                    print("❌ Tester API failed after all retries")
                    break
                
                tester_reply, score, should_continue = result
                
                # Format reply
                if score is not None:
                    if score >= 80:
                        score_badge = f"🟢 **Completeness: {score}/100**"
                    elif score >= 60:
                        score_badge = f"🟡 **Completeness: {score}/100**"
                    elif score >= 40:
                        score_badge = f"🟠 **Completeness: {score}/100**"
                    else:
                        score_badge = f"🔴 **Completeness: {score}/100**"
                    
                    display_reply = f"{tester_reply}\n\n---\n{score_badge}"
                    
                    # Check for early exit
                    if score >= 90:
                        logger.info(f"🎯 HIGH SCORE: {score}/100 >= 90")
                        early_exit = True
                        early_exit_reason = f"High score ({score}/100)"
                    
                    if not should_continue and score >= 80:
                        logger.info(f"✅ COMPLETION: should_continue=false AND score={score}/100 >= 80")
                        early_exit = True
                        early_exit_reason = f"Completion signal (score={score}/100)"
                    
                    if early_exit:
                        logger.info(f"✨ EARLY EXIT: {early_exit_reason}")
                        print(f"\n✨ EARLY EXIT: {early_exit_reason}")
                        print(f"✅ Test case completed successfully!\n")
                else:
                    display_reply = tester_reply
                
                print(f"🧪 Tester reply:\n{display_reply}\n")
                CONVERSATION_LOG.append({"speaker": "Tester", "message": display_reply})
                
                turns_completed = turn
                
                if early_exit:
                    logger.info(f"✅ Completed {turns_completed} turns (early exit)")
                    print(f"✅ Completed {turns_completed} turns (early exit)")
                    break

                # Tester → Mila
                mila_count_before_send = len(page_mila.query_selector_all(mila_sel))
                logger.info(f"Bubble count before send: {mila_count_before_send}")
                
                # ✨ Capture state before sending
                diagnostics.detect_all_chat_elements(page_mila, f"turn{turn}_before_send")
                
                try:
                    send_message_to_mila(page_mila, tester_reply)
                    
                    # ✨ Capture state after sending
                    diagnostics.capture_dom_snapshot(page_mila, f"07_turn{turn}_message_sent")
                    
                    # ✨✨ NEW: Auto-retry loop for technical errors
                    retry_attempts = 0
                    max_auto_retries = 2
                    
                    while retry_attempts < max_auto_retries:
                        # Check for technical error and auto-retry
                        if handle_mila_technical_error(page_mila, max_wait_ms=2000):
                            retry_attempts += 1
                            logger.info(f"🔄 Auto-retry {retry_attempts}/{max_auto_retries}")
                            print(f"🔄 Auto-retry attempt {retry_attempts}/{max_auto_retries}...")
                            
                            # Capture retry state
                            diagnostics.capture_dom_snapshot(page_mila, 
                                f"07_turn{turn}_retry_{retry_attempts}")
                            
                            # Wait for retry to process
                            page_mila.wait_for_timeout(3000)
                        else:
                            # No error detected, proceed
                            logger.debug("✓ No technical error detected")
                            break
                    
                    # Check if error persists after all retries
                    if retry_attempts >= max_auto_retries:
                        logger.error("❌ Technical error persists after max retries")
                        error_state = diagnostics.check_for_errors(page_mila)
                        if error_state["has_errors"]:
                            logger.error(f"   Error: {error_state['error_messages']}")
                            diagnostics.capture_dom_snapshot(page_mila, 
                                f"08_turn{turn}_ERROR_PERSISTENT")
                            take_screenshot(page_mila, f"persistent_error_turn{turn}")
                            print(f"❌ Persistent error after {max_auto_retries} retries")
                            break
                        
                except Exception as e:
                    log_exception(logger, e, "send_message_to_mila")
                    logger.error(f"❌ Failed to send to Mila: {e}")
                    print(f"❌ Failed to send to Mila: {e}")
                    diagnostics.capture_dom_snapshot(page_mila, f"09_turn{turn}_send_FAILED")
                    take_screenshot(page_mila, f"send_failed_turn{turn}")
                    break

                # Wait for Mila's response
                try:
                    new_mila_count = wait_for_new_message(
                        page_mila, mila_sel, mila_count_before_send, timeout_ms=45000
                    )
                    wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=60000)
                except PlaywrightTimeoutError:
                    logger.error("❌ Mila did not reply (timeout)")
                    print("❌ Mila did not reply")
                    
                    # ✨ Capture timeout state for analysis
                    diagnostics.capture_dom_snapshot(page_mila, f"10_turn{turn}_TIMEOUT")
                    diagnostics.detect_all_chat_elements(page_mila, f"turn{turn}_timeout")
                    
                    # ✨ Check if there are error messages
                    error_state = diagnostics.check_for_errors(page_mila)
                    if error_state["has_errors"]:
                        logger.error(f"❌ Errors found during timeout!")
                        logger.error(f"   Error types: {error_state['error_types']}")
                        logger.error(f"   Error messages: {error_state['error_messages']}")
                    
                    take_screenshot(page_mila, f"mila_timeout_turn{turn}")
                    break

                mila_count = new_mila_count
                mila_last = get_mila_last_message_text(page_mila, mila_sel)
                
                if not mila_last:
                    logger.error("❌ Failed to extract Mila's reply")
                    print("❌ Failed to extract Mila's reply")
                    diagnostics.capture_dom_snapshot(page_mila, f"11_turn{turn}_extraction_FAILED")
                    take_screenshot(page_mila, f"extraction_failed_turn{turn}")
                    break
                
                logger.info(f"✓ RECEIVED MILA'S REPLY (Turn {turn})")
                logger.info(f"  Length: {len(mila_last)} chars")
                logger.info(f"  Content: {mila_last[:200]}...")
                print(f"🟠 Mila reply:\n{mila_last}\n")
                CONVERSATION_LOG.append({"speaker": "Mila", "message": mila_last})
                
                # ✨ Capture successful turn completion
                diagnostics.capture_dom_snapshot(page_mila, f"12_turn{turn}_complete")
                diagnostics.detect_all_chat_elements(page_mila, f"turn{turn}_complete")
                take_screenshot(page_mila, f"turn{turn}_complete")

            logger.info("=" * 80)
            logger.info(f"✅ CONVERSATION COMPLETE")
            logger.info(f"  Total turns: {turns_completed}")
            logger.info(f"  Conversation entries: {len(CONVERSATION_LOG)}")
            logger.info("=" * 80)
            print(f"\n{'='*80}")
            print(f"✅ Chat loop finished! Completed {turns_completed} turns")
            print(f"{'='*80}\n")
            
            time.sleep(2)
            
            logger.info("Closing browser")
            browser.close()
            logger.info("✓ Browser closed")

    finally:
        cleanup_tester_session()
        
        # ✨ Always save diagnostics
        try:
            logger.info("Saving diagnostics...")
            diagnostics.save_diagnostics()
            logger.info("✓ Diagnostics saved successfully")
            print("✓ Diagnostics saved")
        except Exception as e:
            logger.error(f"Failed to save diagnostics: {e}")
            print(f"⚠️ Failed to save diagnostics: {e}")

    logger.info(f"Returning: {len(CONVERSATION_LOG)} entries, {turns_completed} turns")
    return CONVERSATION_LOG, turns_completed


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("SCRIPT STARTED FROM COMMAND LINE")
    logger.info("=" * 80)
    logger.info(f"Arguments: {sys.argv}")
    
    try:
        conversation_log, num_turns = run_bridge()
        logger.info("=" * 80)
        logger.info("✅ SCRIPT COMPLETED SUCCESSFULLY")
        logger.info(f"  Conversation entries: {len(conversation_log)}")
        logger.info(f"  Number of turns: {num_turns}")
        logger.info("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Conversation entries: {len(conversation_log)}")
        print(f"   Number of turns: {num_turns}")
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("⚠️ INTERRUPTED BY USER (Ctrl+C)")
        logger.warning("=" * 80)
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        log_exception(logger, e, "main script execution")
        logger.error("=" * 80)
        logger.error(f"❌ SCRIPT FAILED: {type(e).__name__}")
        logger.error(f"   {e}")
        logger.error("=" * 80)
        raise
    finally:
        logger.info("SCRIPT EXECUTION FINISHED")