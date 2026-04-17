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
- COMPLETE RESTART STRATEGY: If retry button appears, restart entire test
"""
import os
import re
import time
import pathlib
import sys
import json
import httpx
from typing import Any, Optional, List, Dict, Tuple
from diagnostic_utils import DiagnosticCapture
from playwright.sync_api import (
    sync_playwright,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)
from dotenv import load_dotenv
from logging_config import setup_logging, log_exception, log_api_call

# =====================================
# CUSTOM EXCEPTIONS
# =====================================

class TechnicalErrorDetected(Exception):
    """Raised when Mila shows a technical error - signals COMPLETE TEST RESTART needed"""
    pass


class RestartTestRequired(Exception):
    """Raised to signal that the entire test case should be restarted from scratch"""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# =====================================
# MILA CHAT METADATA PARSING
# =====================================
# Per the MILA "Website chat - add chat metadata" spec, the LLM may append
# a JSON metadata block wrapped in DOUBLE square brackets to its response,
# e.g.:
#
#     Sure, here is how to do it. [short reply text]
#     [[
#       {"zielerreichung": 8, "risk_level": "low"}
#     ]]
#
# The spec says the metadata must be STRIPPED before the reply reaches the
# end user and kept separately for logging. For our test pipeline that
# means two things:
#   1. The tester bot (which generates the next user message) must receive
#      the CLEAN version, otherwise it'll see the JSON and get confused.
#   2. The metadata should still be captured and surfaced in the Notion
#      conversation page so human reviewers (and KPI calculators) can see
#      what Mila was self-reporting per turn.
#
# This helper finds the first [[ ... ]] block in the text, parses the
# JSON inside, and returns (clean_text, metadata_dict). If no block is
# found or the JSON is invalid, returns (original_text, {}).
_MILA_METADATA_RE = re.compile(r"\[\[\s*(\{.*?\})\s*\]\]", re.DOTALL)


def strip_mila_metadata(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Strip a trailing [[{...}]] metadata block from a Mila reply.

    Returns (clean_text, metadata_dict). metadata_dict is empty when the
    block is missing or the JSON inside isn't parseable. Never raises.
    """
    if not text:
        return text or "", {}
    m = _MILA_METADATA_RE.search(text)
    if not m:
        return text, {}
    try:
        meta = json.loads(m.group(1))
        if not isinstance(meta, dict):
            return text, {}
    except Exception:
        return text, {}
    cleaned = _MILA_METADATA_RE.sub("", text).strip()
    return cleaned, meta


class MilaBackendRejection(RestartTestRequired):
    """
    Raised when Mila's backend itself crashes ("something went terribly
    wrong", "Please try to ask me again") in response to the user message.

    Subclasses RestartTestRequired so all existing catch sites still work,
    but the outer restart loop recognizes this as a CONTENT-INDUCED crash
    rather than a transient environment issue, and caps retries at
    MAX_MILA_REJECTION_RETRIES (default 1) because the same deterministic
    test-case content will simply crash Mila's backend again on the next
    attempt. Burning 5 retries × 2 min each on content the backend can't
    handle wastes time and credits. Observed for safety-adjacent test cases
    where Mila's xi-deepchat API returns HTTP 500 on turn 1.
    """
    pass


class TesterAPIUnavailable(RestartTestRequired):
    """
    Raised when the tester-bot HTTP API (test_bot_headless.py) can't
    produce a valid user message because its upstream OpenAI key is out
    of quota / rate-limited / otherwise unusable.

    Observed on 2026-04-17: every chat.completions.create() returned
    HTTP 429 "You exceeded your current quota". The tester bot caught
    the error and returned it as a reply string starting with
    "❌ OpenAI error:". The bridge (before this fix) treated that as a
    legitimate user message and sent it to Mila, which politely
    explained what 429 errors mean — polluting every conversation in
    the test run with nonsense.

    Restarting the bridge won't fix an OpenAI billing issue, so this
    exception terminates the whole test immediately (retry budget of 0)
    instead of burning the general MAX_TEST_RESTARTS budget. The run
    is marked FAILED and the operator can check OpenAI billing to
    unblock.
    """
    pass


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

# ✅ NEW: Complete restart configuration
MAX_TEST_RESTARTS = int(os.getenv("MAX_TEST_RESTARTS", "5"))  # Max times to restart entire test
RESTART_DELAY = int(os.getenv("RESTART_DELAY", "10"))  # Seconds to wait before restarting

# Separate, tighter retry budget for content-induced Mila backend crashes
# (see MilaBackendRejection). Default 1 means: do ONE retry to rule out a
# transient blip, then bail out immediately. Adversarial test cases that
# deterministically crash Mila's backend should not burn the full 5-retry
# budget — that's ~10 minutes per test case thrown away for no useful signal.
MAX_MILA_REJECTION_RETRIES = int(os.getenv("MAX_MILA_REJECTION_RETRIES", "1"))

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
logger.info(f"  MAX_TEST_RESTARTS: {MAX_TEST_RESTARTS}")
logger.info(f"  RESTART_DELAY: {RESTART_DELAY}s")
logger.info(f"  Test Case: {TEST_CASE}")
logger.info(f"  Persona: {TEST_PERSONA}")
logger.info(f"  Screenshots: {ENABLE_SCREENSHOTS}")
logger.info(f"  Detailed Timing: {DETAILED_TIMING}")
logger.debug(f"  Test Case Details: {len(TEST_CASE_DETAILS)} chars")
logger.debug(f"  Test Case Prompt: {len(TEST_CASE_PROMPT)} chars")
logger.info("=" * 80)

# Conversation log
CONVERSATION_LOG: List[Dict[str, str]] = []

# Per-turn completeness scores for KPI tracking
PER_TURN_SCORES: List[Dict[str, Any]] = []

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
            "#text-input[contenteditable='true']",
            "[contenteditable='true']",
            "input[placeholder='Enter a prompt here']",
            "textarea[placeholder='Enter a prompt here']",
            "div.ai-deepchat--input-container input",
            "div.ai-deepchat--input-container textarea",
            "textarea.ai-deepchat--textarea",
            "input[type='text']:not([name='q'])",
            "div[class*='deepchat'] input",
            "div[class*='deepchat'] textarea",
        ],
        "send_button": (
            "div.submit-button-enlarged, "
            "[role='button'].submit-button-enlarged, "
            "button.ai-deepchat--send, "
            "button[aria-label='Send'], "
            "button:has-text('Send')"
        ),
        # Chat widget structure selectors.
        # The FAB used to be <div class="fab-mode"> but in the current MILA UI
        # it is <button class="fab-mode">. The double selector keeps the fix
        # forward- and backward-compatible. Authoritative open/collapsed state
        # is read from the .chat-container class list (chat-open vs
        # chat-collapsed) by is_mila_chat_open(), not from FAB visibility.
        "chat_container": "div.chat-container",
        "chat_header": "div.chat-header",
        "fab_button": "button.fab-mode, div.fab-mode",
        "header_mode": "div.header-mode",
        "close_button": "button.toggle-icon, span.js-chat-toggle, span.toggle-icon",
        # Legacy header selector (kept for backward compatibility)
        "header": "div.chat-header",
        # Reset/clear chat
        "reset_chat": "div.reset-chat",
        "clear_history": "div.reset-chat, a.clear-history, a[class*='clear-history'], a.chat-dropdown-link",
        # Menu (kebab icon in header)
        "menu_opener": "qz-dropdown qz-icon[slot='opener']",
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
            reply_lower = reply.lower()

            # Persistent upstream OpenAI failures — no point retrying, no
            # point passing this to Mila as a "user message". Abort the
            # whole test immediately with a clear cause; the operator has
            # to fix OpenAI billing / the key before anything will work.
            persistent_upstream = any(
                s in reply_lower
                for s in (
                    "exceeded your current quota",
                    "insufficient_quota",
                    "billing",
                    "invalid_api_key",
                    "too many consecutive errors",
                )
            )
            if persistent_upstream:
                logger.error(
                    "🚨 Tester API upstream (OpenAI) is persistently failing — "
                    "aborting test immediately. No retry will help."
                )
                print(
                    "🚨 Tester API upstream is down (OpenAI quota/billing). "
                    "Fix the OpenAI account and rerun."
                )
                raise TesterAPIUnavailable(
                    f"Tester API returned persistent upstream error: {reply[:200]}"
                )

            # Transient issues (network timeout, temporary rate limit) —
            # retry inside this request's budget.
            if "timeout" in reply_lower or "rate limit" in reply_lower:
                if retry_count < MAX_RETRIES - 1:
                    logger.info(f"⏳ Retrying after {RETRY_DELAY}s...")
                    print(f"⏳ Retrying after {RETRY_DELAY}s (attempt {retry_count + 2}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                    return send_to_tester_api(message, retry_count + 1)

            # Any OTHER ❌-prefixed reply we don't recognize: don't send
            # garbage to Mila either. Signal failure to the caller with
            # None; the turn loop's `if result is None` branch will raise
            # a restart rather than silently pretending the test succeeded.
            logger.error(f"❌ Unrecognized tester error — returning None to caller")
            return None

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
    NOW WITH: Continuous error checking during the wait - RAISES RestartTestRequired if error found.
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
        
        # ✅✅✅ CRITICAL: Check for technical errors DURING the wait
        # If error found, RAISE EXCEPTION TO RESTART ENTIRE TEST
        if check_count % 3 == 0:  # Check every 3rd iteration
            try:
                # Quick check for error message (no long timeout)
                # Use partial text matching with :has-text() for robustness.
                # These patterns specifically indicate Mila's xi-deepchat API
                # returned HTTP 500 (content-induced backend crash). They are
                # raised as MilaBackendRejection so the outer restart loop
                # can cap retries tightly instead of burning the full budget
                # on content that will deterministically crash again.
                error_text_patterns = [
                    "something went terribly wrong",
                    "Please try to ask me again",
                    "Error, please try again",
                ]

                for pattern in error_text_patterns:
                    try:
                        # Use :has-text() for partial matching (more robust than exact text match)
                        error_elem = page.locator(f":has-text('{pattern}')").first
                        if error_elem.is_visible(timeout=100):
                            logger.error(f"🚨 MILA BACKEND REJECTION during wait (after {elapsed:.1f}s)")
                            logger.error(f"   Pattern matched: '{pattern}'")
                            logger.error(f"   ⚠️ Content likely triggers xi-deepchat 500 — will retry at most {MAX_MILA_REJECTION_RETRIES} time(s)")
                            print(f"🚨 Mila backend rejected this content — fast-fail path")

                            # Take screenshot of error
                            take_screenshot(page, f"error_detected_restart_needed")

                            # Raise the specific subclass so run_bridge_with_restart
                            # knows to use the tighter MAX_MILA_REJECTION_RETRIES
                            # budget instead of MAX_TEST_RESTARTS.
                            raise MilaBackendRejection(
                                f"Mila backend rejection pattern '{pattern}' "
                                f"detected at {elapsed:.1f}s into wait"
                            )

                    except PlaywrightTimeoutError:
                        # Error selector not found, continue
                        continue
                    except RestartTestRequired:
                        # Re-raise to propagate up (covers both RestartTestRequired
                        # and its MilaBackendRejection subclass)
                        raise
                    except Exception as e:
                        # Other errors, ignore and continue
                        logger.debug(f"Error pattern '{pattern}' check failed: {e}")
                        continue
            except RestartTestRequired:
                # Propagate to caller
                raise
            except Exception as e:
                logger.debug(f"Error check exception: {e}")
        
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

        # Wait for page to settle after login (navigation + async loading)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
            logger.debug("  Network idle after login")
        except Exception:
            logger.debug("  Network idle timeout, continuing...")
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


def get_mila_chat_container_classes(page: Page) -> str:
    """
    Return the full className string of the Mila chat container, or "" if
    the container is not in the DOM yet.

    The container's class list is the authoritative source of chat state:
        .chat-container.chat-open       → widget expanded, input visible
        .chat-container.chat-collapsed  → widget minimized, input hidden

    All previous indirect checks (header-mode visibility, FAB visibility,
    aria-expanded attributes) can lie during transitional states, so we
    always consult this instead.
    """
    try:
        return page.evaluate(
            "() => { const c = document.querySelector('div.chat-container'); "
            "return c ? c.className : ''; }"
        ) or ""
    except Exception as e:
        logger.debug(f"  get_mila_chat_container_classes failed: {e}")
        return ""


def is_mila_chat_open(page: Page) -> bool:
    """Return True iff the Mila chat container is currently in the open state."""
    classes = get_mila_chat_container_classes(page)
    # "chat-open" is an explicit positive signal. If neither class is present
    # we can't say for sure, so err on the side of "not open" and let the
    # caller click the FAB.
    return "chat-open" in classes and "chat-collapsed" not in classes


def wait_for_mila_chat_state(page: Page, want_open: bool, timeout_ms: int = 5000) -> bool:
    """
    Wait for the chat container to reach the requested state. Returns True
    on success, False on timeout. Does not raise.
    """
    want_class = "chat-open" if want_open else "chat-collapsed"
    try:
        page.wait_for_function(
            f"""() => {{
                const c = document.querySelector('div.chat-container');
                return c && c.classList.contains('{want_class}');
            }}""",
            timeout=timeout_ms,
        )
        return True
    except Exception as e:
        logger.debug(f"  wait_for_mila_chat_state({want_open}) timed out: {e}")
        return False


def open_mila_chat(page: Page):
    """
    Open the Mila chat widget if it is not already open.

    State is read from the .chat-container class list (chat-open vs
    chat-collapsed), not from FAB / header visibility, because those elements
    remain in the DOM in both states and their individual CSS visibility can
    lag behind the transition.
    """
    logger.info("=" * 60)
    logger.info("OPENING MILA CHAT WIDGET")
    logger.info("=" * 60)

    try:
        # Wait briefly for the container to appear (page just loaded)
        try:
            page.locator(SELECTORS["mila"]["chat_container"]).first.wait_for(
                state="attached", timeout=5000
            )
        except Exception:
            logger.warning("⚠️ Chat container not yet attached — proceeding anyway")

        if is_mila_chat_open(page):
            logger.info("💬 Chat widget already open (chat-container.chat-open)")
            print("💬 Mila chat widget already open.")
            logger.info("=" * 60)
            take_screenshot(page, "chat_already_open")
            return

        classes = get_mila_chat_container_classes(page)
        logger.info(f"💬 Chat container state: '{classes}' — needs opening")
        print("💬 Opening Mila chat widget...")

        # Click the FAB button. Current MILA UI uses <button class="fab-mode">;
        # the selector also matches the older <div class="fab-mode"> for
        # backward compatibility.
        fab = page.locator(SELECTORS["mila"]["fab_button"]).first
        try:
            fab.wait_for(state="visible", timeout=10000)
        except Exception:
            # FAB not visible? Use JavaScript to click it anyway — it's
            # attached to the DOM either way, and in some transitional states
            # Playwright's visibility check is too strict.
            logger.warning("⚠️ FAB button not 'visible' to Playwright — using JS click fallback")
            clicked = page.evaluate(
                """() => {
                    const b = document.querySelector('button.fab-mode, div.fab-mode');
                    if (b) { b.click(); return true; }
                    return false;
                }"""
            )
            if not clicked:
                logger.error("❌ FAB button not found in DOM — cannot open chat")
                raise RuntimeError("Mila FAB button not found in DOM")
        else:
            fab.click()

        page.wait_for_timeout(300)

        if wait_for_mila_chat_state(page, want_open=True, timeout_ms=5000):
            logger.info("✓ Chat widget opened successfully (chat-container.chat-open)")
            print("✓ Mila chat widget opened")
        else:
            new_classes = get_mila_chat_container_classes(page)
            logger.error(
                f"❌ FAB clicked but chat-container did not reach chat-open state "
                f"(current classes: '{new_classes}')"
            )
            print(f"❌ Could not confirm Mila chat opened. Classes: {new_classes}")
            raise RuntimeError(
                f"Mila chat did not transition to chat-open after FAB click "
                f"(classes: {new_classes})"
            )

        logger.info("=" * 60)
        take_screenshot(page, "chat_opened")

    except Exception as e:
        log_exception(logger, e, "open_mila_chat")
        logger.warning("=" * 60)
        print(f"ℹ️ Could not open Mila chat widget automatically: {e}")
        # Do NOT silently continue — let subsequent operations fail loudly
        # so RestartTestRequired kicks in and this gets retried in a fresh
        # browser context instead of the old broken state.
        raise


def clear_mila_history(page: Page):
    """Clear Mila chat history using 'Reset chat' from the kebab menu"""
    logger.info("=" * 60)
    logger.info("CLEARING MILA CHAT HISTORY")
    logger.info("=" * 60)

    try:
        print("🗑️  Clearing Mila chat history...")

        # Step 1: Open the kebab menu (three-dot icon) in the header
        menu_opened = False

        # Primary: use the qz-dropdown opener in header-mode
        menu_selectors = [
            SELECTORS["mila"]["menu_opener"],          # qz-dropdown qz-icon[slot='opener']
            "div.header-mode qz-icon[slot='opener']",  # more specific
            "qz-icon[name='more-vertical']",           # by icon name
            "div.header-mode qz-dropdown",             # the dropdown itself
            # Legacy fallbacks
            "svg.chevron-icon",
            "svg#chevron",
            "button[aria-label*='menu']",
            "button[aria-label*='Menu']",
            ".ai-deepchat--header svg",
            ".ai-deepchat--header button",
            "[class*='menu-icon']",
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

        # Step 2: Click "Reset chat" (new name) or "Clear History" (legacy)
        clear_selectors = [
            SELECTORS["mila"]["reset_chat"],           # div.reset-chat
            "div.reset-chat span",                     # the span inside reset-chat
            ":has-text('Reset chat')",                 # text match
            "a.clear-history",
            "a[class*='clear-history']",
            "a.chat-dropdown-link:has-text('Clear History')",
            "a:has-text('Clear History')",
            "button:has-text('Clear History')",
        ]

        clicked = False
        logger.debug("Looking for Reset chat / Clear History button")
        for clear_sel in clear_selectors:
            try:
                logger.debug(f"  Trying selector: {clear_sel}")
                clear_btn = page.locator(clear_sel).first
                if clear_btn.is_visible(timeout=2000):
                    logger.info(f"✓ Found Reset/Clear button: {clear_sel}")
                    print(f"   Found Reset/Clear with selector: {clear_sel}")
                    clear_btn.click()
                    logger.info("✓ Clicked 'Reset chat'")
                    print("   ✓ Clicked 'Reset chat'")
                    page.wait_for_timeout(1500)
                    clicked = True
                    break
            except Exception as e:
                logger.debug(f"  Selector failed: {e}")
                continue

        if not clicked:
            # Try text match as last resort
            for text_label in ["Reset chat", "Clear History"]:
                try:
                    logger.debug(f"Trying text match for '{text_label}'")
                    text_link = page.get_by_text(text_label, exact=False)
                    if text_link.is_visible(timeout=2000):
                        text_link.click()
                        logger.info(f"✓ Clicked via text match: '{text_label}'")
                        print(f"   ✓ Clicked '{text_label}' using text match")
                        page.wait_for_timeout(1500)
                        clicked = True
                        break
                except Exception as e:
                    logger.debug(f"  Text match failed: {e}")

        if not clicked:
            logger.error("❌ Could not find 'Reset chat' or 'Clear History' button")
            print("   ⚠️ Could not find 'Reset chat' or 'Clear History' button")
            logger.info("=" * 60)
            return

        # Step 3: Confirm clearing if a confirmation dialog appears
        try:
            logger.debug("Looking for confirmation button")
            confirm_selectors = [
                "button:has-text('Confirm')",
                "button:has-text('Yes')",
                "button:has-text('OK')",
                "button:has-text('Clear')",
                "button:has-text('Reset')",
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
    """
    Ensure Mila chat is in the open state. Uses the authoritative
    .chat-container class list (chat-open / chat-collapsed) rather than
    FAB visibility, which lies during transitional states.

    Raises RuntimeError if the chat cannot be opened after a FAB click +
    JS-click fallback. The caller's except-Exception handler will convert
    that into a RestartTestRequired so the whole test restarts in a fresh
    browser context (which is the only reliable recovery for the MILA
    widget's known zombie state).
    """
    logger.debug("Ensuring Mila chat is open")

    try:
        if is_mila_chat_open(page):
            logger.debug("  Chat already open (chat-container.chat-open)")
        else:
            classes_before = get_mila_chat_container_classes(page)
            logger.info(
                f"💬 Chat not open — container classes: '{classes_before}' — reopening"
            )
            print("💬 Reopening Mila chat widget...")

            # Try a normal Playwright click first; fall back to a raw JS
            # click if Playwright refuses (transitional visibility issues).
            fab = page.locator(SELECTORS["mila"]["fab_button"]).first
            clicked = False
            try:
                fab.click(timeout=3000)
                clicked = True
            except Exception as click_err:
                logger.warning(
                    f"  FAB Playwright click failed ({click_err}) — trying JS fallback"
                )
                clicked = page.evaluate(
                    """() => {
                        const b = document.querySelector('button.fab-mode, div.fab-mode');
                        if (b) { b.click(); return true; }
                        return false;
                    }"""
                )

            if not clicked:
                raise RuntimeError("FAB button not found in DOM during ensure_mila_chat_open")

            # Wait for the container class to flip to chat-open — this is
            # the only reliable "we really opened" signal.
            if not wait_for_mila_chat_state(page, want_open=True, timeout_ms=5000):
                classes_after = get_mila_chat_container_classes(page)
                raise RuntimeError(
                    f"FAB clicked but chat-container did not reach chat-open state "
                    f"(classes: '{classes_after}')"
                )
            logger.info("  ✓ Chat reopened successfully")

        # Scroll chat to bottom so new messages are visible. This must run
        # regardless of how we got here.
        page.evaluate("""
            // Try deep-chat shadow DOM first
            const deepChat = document.querySelector('deep-chat');
            if (deepChat && deepChat.shadowRoot) {
                const messages = deepChat.shadowRoot.querySelector('#messages');
                if (messages) {
                    messages.scrollTop = messages.scrollHeight;
                }
            }
            // Fallback: try regular DOM selectors
            const chatContainer = document.querySelector('div.ai-deepchat--body, div[class*="chat-messages"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        """)
        page.wait_for_timeout(300)
        logger.debug("  Chat scrolled to bottom")

    except Exception as e:
        # Loud failure — propagate to send_message_to_mila, which will retry
        # up to its own max_retries, then raise, which our new
        # PlaywrightTimeoutError / generic-exception handlers in the main
        # turn loop convert into RestartTestRequired.
        logger.error(f"❌ ensure_mila_chat_open failed: {e}")
        raise


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
                            // Clear existing content
                            element.innerHTML = '';
                            element.textContent = {repr(text)};
                            // Fire multiple events to ensure deep-chat picks up the change
                            element.dispatchEvent(new Event('input', {{ bubbles: true, composed: true }}));
                            element.dispatchEvent(new Event('change', {{ bubbles: true, composed: true }}));
                            element.dispatchEvent(new KeyboardEvent('keyup', {{ bubbles: true, composed: true }}));
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
                try:
                    box.fill("")
                except Exception:
                    pass
                box.click()
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
                logger.warning("  Send button not visible, trying Enter key")
                try:
                    # Focus the input first, then press Enter
                    box.click()
                    page.wait_for_timeout(200)
                    page.keyboard.press("Enter")
                    logger.debug("  Enter key pressed")
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

def run_bridge_single_attempt() -> Tuple[List[Dict[str, str]], int]:
    """
    Run a SINGLE attempt of the headless bridge.
    Raises RestartTestRequired if technical error detected.
    Returns: (conversation_log, number_of_turns)
    """
    global CONVERSATION_LOG, PER_TURN_SCORES
    CONVERSATION_LOG = []
    PER_TURN_SCORES = []
    
    turns_completed = 0

    # ✨ Initialize diagnostics
    diagnostics = DiagnosticCapture(output_dir="diagnostics")
    logger.info("✓ Diagnostics system initialized")

    logger.info("=" * 80)
    logger.info("STARTING HEADLESS BRIDGE BOT - SINGLE ATTEMPT")
    logger.info("=" * 80)
    
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
            
            # Strip any [[{...}]] metadata block Mila may append per the
            # "Website chat - add chat metadata" spec. The clean version is
            # what we show the tester bot and persist in the transcript;
            # the raw metadata is logged separately for KPI + debugging.
            mila_last, mila_meta = strip_mila_metadata(mila_last)
            if mila_meta:
                meta_json = json.dumps(mila_meta, ensure_ascii=False)
                logger.info(f"🏷️ Mila metadata (first message): {meta_json}")
                print(f"🏷️ Mila metadata: {meta_json}")

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
                    logger.error("❌ Tester API failed after all retries")
                    print("❌ Tester API failed after all retries")
                    # Do NOT silently break — that would let the outer
                    # handler report the partial run as success. A dead
                    # tester API means we have no valid user message to
                    # send, so this whole test attempt is junk. Raising
                    # RestartTestRequired surfaces it as an honest failure
                    # and lets the restart strategy retry in case it was
                    # a transient network issue. Persistent OpenAI
                    # billing/quota issues are caught earlier as
                    # TesterAPIUnavailable and terminate the test
                    # immediately (no restart).
                    raise RestartTestRequired(
                        f"Tester API returned no valid reply after "
                        f"{MAX_RETRIES} retries on turn {turn}"
                    )
                
                tester_reply, score, should_continue = result

                # Track per-turn completeness score for KPI calculation
                PER_TURN_SCORES.append({
                    "turn": turn,
                    "score": score,
                    "should_continue": should_continue,
                })

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
                    diagnostics.capture_dom_snapshot(page_mila, f"07_turn{turn}_message_sent")
                    
                    # ✅ NEW: Simple wait with error detection
                    # If RestartTestRequired is raised, it propagates up
                    new_mila_count = wait_for_new_message(
                        page_mila, mila_sel, mila_count_before_send, timeout_ms=45000
                    )
                    wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=60000)
                    
                    mila_count = new_mila_count
                    logger.info("✓ Response received successfully")
                        
                except RestartTestRequired:
                    # Error detected - propagate to trigger restart
                    logger.error("🚨 RestartTestRequired raised - propagating up")
                    raise
                    
                except PlaywrightTimeoutError as e:
                    # Mila did not respond in time — this is a REAL failure.
                    # Previously this block silently `break`'d out of the loop, leaving
                    # the outer harness to report the run as "COMPLETED SUCCESSFULLY" with
                    # whatever partial turns had accumulated. That was poisoning KPIs with
                    # 1-turn "conversations" where Mila never replied (see Issue #1 in the
                    # fix list). Now we raise RestartTestRequired so the restart strategy
                    # retries the test from scratch, and if MAX_TEST_RESTARTS is exceeded
                    # the script exits non-zero and `run_test_executions.py` marks the
                    # execution as failed rather than "Test Executed".
                    logger.error("❌ Mila did not reply (timeout)")
                    print("❌ Mila did not reply (timeout)")

                    diagnostics.capture_dom_snapshot(page_mila, f"10_turn{turn}_TIMEOUT")
                    diagnostics.detect_all_chat_elements(page_mila, f"turn{turn}_timeout")

                    error_state = diagnostics.check_for_errors(page_mila)
                    if error_state["has_errors"]:
                        logger.error(f"❌ Errors found during timeout!")
                        logger.error(f"   Error types: {error_state['error_types']}")
                        logger.error(f"   Error messages: {error_state['error_messages']}")

                    take_screenshot(page_mila, f"mila_timeout_turn{turn}")
                    raise RestartTestRequired(
                        f"Mila did not reply within timeout on turn {turn} "
                        f"(waited for new bubble / typing-complete); treating as failure."
                    ) from e

                except Exception as e:
                    log_exception(logger, e, "send_message_to_mila or wait")
                    logger.error(f"❌ Failed: {e}")
                    print(f"❌ Failed: {e}")
                    diagnostics.capture_dom_snapshot(page_mila, f"09_turn{turn}_FAILED")
                    take_screenshot(page_mila, f"failed_turn{turn}")
                    # Same rationale as the PlaywrightTimeoutError handler above: don't
                    # mask errors with a silent break — escalate to a restart.
                    raise RestartTestRequired(
                        f"Unexpected error on turn {turn}: {type(e).__name__}: {e}"
                    ) from e

                # Extract Mila's response
                mila_last = get_mila_last_message_text(page_mila, mila_sel)

                if not mila_last:
                    logger.error("❌ Failed to extract Mila's reply")
                    print("❌ Failed to extract Mila's reply")
                    diagnostics.capture_dom_snapshot(page_mila, f"11_turn{turn}_extraction_FAILED")
                    take_screenshot(page_mila, f"extraction_failed_turn{turn}")
                    # Empty reply = broken widget state (e.g. the "Calling RAG/Vector
                    # Search" zombie state observed in the frontend). Restart the test
                    # instead of pretending it succeeded.
                    raise RestartTestRequired(
                        f"Could not extract Mila's reply on turn {turn} "
                        f"(widget likely in broken state)"
                    )
                
                # Strip + capture any [[{...}]] metadata block per the
                # "Website chat - add chat metadata" spec. Clean text goes
                # to the tester bot and to the transcript; metadata is
                # logged to stdout so format_conversation_page can render it
                # as part of the Mila message in Notion.
                mila_last, mila_meta = strip_mila_metadata(mila_last)
                if mila_meta:
                    meta_json = json.dumps(mila_meta, ensure_ascii=False)
                    logger.info(f"🏷️ Mila metadata (Turn {turn}): {meta_json}")
                    print(f"🏷️ Mila metadata: {meta_json}")

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

    except RestartTestRequired:
        # Propagate to outer handler
        logger.error("🚨 RestartTestRequired - cleaning up and propagating")
        raise
        
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


def run_bridge_with_restart() -> Tuple[List[Dict[str, str]], int]:
    """
    Run the bridge with a complete restart strategy.

    Two separate retry budgets are tracked:

      * restart_attempt     — for general RestartTestRequired errors
                              (transient Playwright / environment issues).
                              Capped at MAX_TEST_RESTARTS (default 5).

      * mila_reject_attempt — for MilaBackendRejection specifically, i.e.
                              Mila's xi-deepchat API returning the
                              "something went terribly wrong" error bubble
                              (HTTP 500 induced by the test-case content).
                              Capped at MAX_MILA_REJECTION_RETRIES
                              (default 1 — one retry to rule out transience,
                              then fail fast). Burning the full 5-retry
                              budget on content the backend deterministically
                              can't handle wastes ~10 minutes per test case.

    Both counters can independently reach their cap. Either one triggering
    the cap terminates the test with ([], 0), which propagates to exit 1.
    """
    restart_attempt = 0
    mila_reject_attempt = 0

    while True:
        attempt_number = restart_attempt + mila_reject_attempt + 1
        total_budget = MAX_TEST_RESTARTS + MAX_MILA_REJECTION_RETRIES + 1
        try:
            logger.info("=" * 100)
            if restart_attempt + mila_reject_attempt == 0:
                logger.info(
                    f"STARTING TEST EXECUTION (Attempt 1/{total_budget})"
                )
            else:
                logger.info(
                    f"RESTARTING TEST EXECUTION "
                    f"(Attempt {attempt_number}/{total_budget}, "
                    f"general={restart_attempt}/{MAX_TEST_RESTARTS}, "
                    f"mila-reject={mila_reject_attempt}/{MAX_MILA_REJECTION_RETRIES})"
                )
            logger.info("=" * 100)

            if restart_attempt + mila_reject_attempt > 0:
                print(f"\n{'='*100}")
                print(f"🔄 RESTARTING ENTIRE TEST - Attempt {attempt_number}/{total_budget}")
                print(f"Previous attempt failed due to technical error")
                print(f"Waiting {RESTART_DELAY} seconds before restart...")
                print(f"{'='*100}\n")
                time.sleep(RESTART_DELAY)

            # Emit a boundary marker that format_conversation_page.py uses to
            # scope the conversation-page parser to only the FINAL attempt's
            # output. Without this, stdout captured by run_test_executions.py's
            # subprocess.run(capture_output=True) contains concatenated output
            # from all restart attempts, and the Notion conversation page ends
            # up showing, e.g., "Turn 1, Turn 2, Turn 3, Turn 4, Turn 5, Turn 1"
            # (turns from a failed attempt 1 followed by turn 1 of attempt 2).
            # The parser does `text.rfind(boundary)` and keeps only the tail,
            # so each attempt overwrites the previous one in rendering terms
            # while the raw bridge log still has every attempt for forensics.
            print("\n===BRIDGE_ATTEMPT_BOUNDARY===\n")

            # Run single attempt - if successful, return results
            conversation_log, turns = run_bridge_single_attempt()

            # If we got here, test completed successfully!
            logger.info("=" * 100)
            logger.info("✅ TEST COMPLETED SUCCESSFULLY - NO ERRORS")
            logger.info(f"  Turns completed: {turns}")
            logger.info(f"  General restarts: {restart_attempt}")
            logger.info(f"  Mila-reject restarts: {mila_reject_attempt}")
            logger.info("=" * 100)

            print(f"\n{'='*100}")
            print(f"✅ TEST COMPLETED SUCCESSFULLY!")
            print(f"  Turns: {turns}")
            print(f"  Total restart attempts: {restart_attempt + mila_reject_attempt}")
            print(f"{'='*100}\n")

            return conversation_log, turns

        except TesterAPIUnavailable as e:
            # OpenAI / tester-bot infra is down. No retry will fix this —
            # the operator has to fix billing/key first. Abort immediately
            # with a distinct failure reason so run_test_executions marks
            # the execution as [FAILED] and we don't waste the rest of
            # the queue.
            logger.error("=" * 100)
            logger.error(f"🚨 TESTER API UNAVAILABLE: {e.reason}")
            logger.error(
                "   No retry — upstream OpenAI issue (quota/billing/auth). "
                "Fix the OpenAI account and re-run."
            )
            logger.error("=" * 100)
            print(f"\n{'='*100}")
            print("❌ TESTER API UNAVAILABLE — upstream OpenAI issue")
            print("   Check OpenAI billing / key, then re-trigger the test run")
            print(f"{'='*100}\n")
            print("BRIDGE_FAILURE_REASON=tester_api_unavailable")
            return [], 0

        except MilaBackendRejection as e:
            # Content-induced backend crash. Use the tight retry budget —
            # retrying does not help when the same deterministic input will
            # crash Mila's backend the same way next time.
            mila_reject_attempt += 1
            logger.error("=" * 100)
            logger.error(f"🚨 MILA BACKEND REJECTION: {e.reason}")
            logger.error(
                f"   Mila-reject attempt: "
                f"{mila_reject_attempt}/{MAX_MILA_REJECTION_RETRIES}"
            )
            logger.error("=" * 100)

            if mila_reject_attempt > MAX_MILA_REJECTION_RETRIES:
                logger.error("=" * 100)
                logger.error("❌ MILA BACKEND REJECTED TEST CONTENT - fail fast")
                logger.error(
                    f"   Exhausted {MAX_MILA_REJECTION_RETRIES} mila-reject retries — "
                    f"test content appears to deterministically crash Mila's xi-deepchat API."
                )
                logger.error("   This is NOT a bot bug. Report the test case to the MILA team.")
                logger.error("=" * 100)
                print(f"\n{'='*100}")
                print(f"❌ MILA BACKEND REJECTED CONTENT — fail fast")
                print(f"   Retries exhausted: {mila_reject_attempt}/{MAX_MILA_REJECTION_RETRIES + 1}")
                print(f"   This is a MILA backend bug, not a bot bug.")
                print(f"{'='*100}\n")
                # Sentinel line in stdout so run_test_executions.py (and anyone
                # tailing logs) can distinguish "content crashed Mila" from
                # "generic bridge failure" without parsing reason strings.
                print("BRIDGE_FAILURE_REASON=mila_backend_rejection")
                return [], 0

            # Continue to next iteration (one retry to rule out transient blip)
            continue

        except RestartTestRequired as e:
            restart_attempt += 1
            logger.error("=" * 100)
            logger.error(f"🚨 RESTART REQUIRED: {e.reason}")
            logger.error(f"   General restart attempt: {restart_attempt}/{MAX_TEST_RESTARTS}")
            logger.error("=" * 100)

            if restart_attempt > MAX_TEST_RESTARTS:
                logger.error("❌ MAX RESTARTS EXCEEDED - TEST FAILED")
                print(f"\n{'='*100}")
                print(f"❌ MAX RESTARTS EXCEEDED ({MAX_TEST_RESTARTS})")
                print(f"Test failed after {restart_attempt} general restart attempts")
                print(f"{'='*100}\n")
                print("BRIDGE_FAILURE_REASON=max_restarts_exceeded")
                return [], 0

            # Continue to next iteration (restart)
            continue

        except Exception as e:
            # Other unexpected errors
            log_exception(logger, e, "run_bridge_with_restart")
            logger.error("=" * 100)
            logger.error(f"❌ UNEXPECTED ERROR: {type(e).__name__}")
            logger.error(f"   {e}")
            logger.error("=" * 100)
            print(f"\n❌ Unexpected error: {e}")
            print("BRIDGE_FAILURE_REASON=unexpected_error")
            return [], 0


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("SCRIPT STARTED FROM COMMAND LINE")
    logger.info("=" * 80)
    logger.info(f"Arguments: {sys.argv}")
    logger.info(f"Restart strategy: ENABLED (max {MAX_TEST_RESTARTS} restarts)")
    
    try:
        conversation_log, num_turns = run_bridge_with_restart()
        
        if num_turns > 0:
            logger.info("=" * 80)
            logger.info("✅ SCRIPT COMPLETED SUCCESSFULLY")
            logger.info(f"  Conversation entries: {len(conversation_log)}")
            logger.info(f"  Number of turns: {num_turns}")
            logger.info(f"  Per-turn scores: {PER_TURN_SCORES}")
            logger.info("=" * 80)
            print(f"\n📊 Summary:")
            print(f"   Conversation entries: {len(conversation_log)}")
            print(f"   Number of turns: {num_turns}")

            # Output per-turn scores as parseable line for run_test_executions.py
            score_values = [s.get("score") for s in PER_TURN_SCORES if s.get("score") is not None]
            if score_values:
                print(f"   Per-turn scores: {json.dumps(score_values)}")
        else:
            logger.error("=" * 80)
            logger.error("❌ SCRIPT FAILED")
            logger.error("  Test did not complete successfully")
            logger.error("=" * 80)
            print(f"\n❌ Test failed to complete")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("⚠️ INTERRUPTED BY USER (Ctrl+C)")
        logger.warning("=" * 80)
        print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_exception(logger, e, "main script execution")
        logger.error("=" * 80)
        logger.error(f"❌ SCRIPT FAILED: {type(e).__name__}")
        logger.error(f"   {e}")
        logger.error("=" * 80)
        raise
    finally:
        logger.info("SCRIPT EXECUTION FINISHED")