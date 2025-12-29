# playwright_bridge_bot_headless.py
"""
Headless version of the Playwright bridge.
Uses HTTP API instead of Streamlit for the tester bot.

MODIFICATIONS:
- Tracks number of turns
- Breaks loop early if completeness score >= 85
"""
import os
import time
import pathlib
import sys
import httpx
from typing import Optional, List, Dict, Tuple

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
TESTER_API_URL = os.getenv("TESTER_API_URL", "http://localhost:8501")  # HTTP API endpoint
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

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

logger.info(f"Configuration loaded:")
logger.info(f"  MILA_URL: {MILA_URL}")
logger.info(f"  TESTER_API_URL: {TESTER_API_URL}")
logger.info(f"  MAX_TURNS: {MAX_TURNS}")
logger.info(f"  Test Case: {TEST_CASE}")
logger.info(f"  Persona: {TEST_PERSONA}")
logger.debug(f"  Test Case Details: {TEST_CASE_DETAILS[:100]}...")
logger.debug(f"  Test Case Prompt: {len(TEST_CASE_PROMPT)} chars")

# Conversation log
CONVERSATION_LOG: List[Dict[str, str]] = []

# Session ID for the test bot API
tester_session_id: Optional[str] = None


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


# =====================================
# TESTER BOT HTTP API FUNCTIONS
# =====================================

def create_tester_session() -> Optional[str]:
    """Create a new test session via HTTP API"""
    global tester_session_id
    
    logger.info("Creating tester session via HTTP API")
    logger.debug(f"  Test case: {TEST_CASE}")
    logger.debug(f"  Persona: {TEST_PERSONA}")
    
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
        
        log_api_call(logger, "POST", f"{TESTER_API_URL}/session/create", response.status_code, duration)
        
        response.raise_for_status()
        data = response.json()
        tester_session_id = data.get("session_id")
        
        logger.info(f"✓ Created tester session: {tester_session_id}")
        return tester_session_id
        
    except Exception as e:
        log_exception(logger, e, "create_tester_session")
        logger.error(f"❌ Error creating tester session: {e}")
        return None


def send_to_tester_api(message: str) -> Optional[Tuple[str, Optional[int], bool]]:
    """
    Send message to tester bot via HTTP API.
    
    Returns: (reply, score, should_continue) or None on error
    """
    if not tester_session_id:
        logger.error("❌ No active tester session")
        print("❌ No active tester session")
        return None
    
    logger.info(f"Sending message to tester API (session: {tester_session_id})")
    logger.debug(f"  Message: {message[:100]}...")
    
    try:
        start_time = time.time()
        response = httpx.post(
            f"{TESTER_API_URL}/session/{tester_session_id}/message",
            json={"message": message},
            timeout=120.0  # Long timeout for OpenAI calls
        )
        duration = time.time() - start_time
        
        log_api_call(logger, "POST", f"{TESTER_API_URL}/session/{tester_session_id}/message", 
                    response.status_code, duration)
        
        response.raise_for_status()
        data = response.json()
        reply = data.get("reply", "")
        score = data.get("score")
        should_continue = data.get("should_continue", True)  # ✅ Extract should_continue
        
        logger.info(f"✓ Received reply from tester API")
        logger.debug(f"  Score: {score}")
        logger.debug(f"  Should continue: {should_continue}")
        logger.debug(f"  Reply: {reply[:100]}...")
        
        return reply, score, should_continue
        
    except Exception as e:
        log_exception(logger, e, "send_to_tester_api")
        logger.error(f"❌ Error sending to tester API: {e}")
        print(f"❌ Error sending to tester API: {e}")
        return None


# =====================================
# MILA HELPERS (same as before)
# =====================================

def wait_for_mila_typing_complete(page: Page, selector: str, timeout_ms: int = 60000):
    """Wait for Mila to finish typing"""
    logger.debug(f"Waiting for Mila to finish typing (timeout: {timeout_ms}ms)")
    print("⏳ Waiting for Mila to finish typing...")
    
    end_time = time.time() + (timeout_ms / 1000)
    last_text = None
    stable_count = 0
    
    while time.time() < end_time:
        try:
            elems = page.query_selector_all(selector)
            if elems:
                last_elem = elems[-1]
                current_text = last_elem.inner_text().strip()
                
                if current_text in ("...", "• • •", ""):
                    last_text = None
                    stable_count = 0
                    time.sleep(0.5)
                    continue
                
                if current_text == last_text:
                    stable_count += 1
                    if stable_count >= 2:
                        logger.info("✓ Mila finished typing")
                        print("✓ Mila finished typing")
                        return
                else:
                    stable_count = 0
                    last_text = current_text
                
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error checking typing status: {e}")
            print(f"   ⚠️ Error checking typing status: {e}")
            time.sleep(0.5)
    
    logger.warning("⚠️ Typing wait timeout, proceeding anyway...")
    print("⚠️ Typing wait timeout, proceeding anyway...")


def get_mila_last_message_text(page: Page, selector: str, retry_count: int = 3) -> Optional[str]:
    """Get the last message text from Mila"""
    for attempt in range(retry_count):
        try:
            elems = page.query_selector_all(selector)
            if not elems:
                if attempt < retry_count - 1:
                    print(f"   ⚠️ No message elements found, retry {attempt + 1}/{retry_count}")
                    time.sleep(1)
                    continue
                return None
            
            last_elem = elems[-1]
            time.sleep(0.5)
            
            try:
                text = last_elem.inner_text().strip()
                if text and text not in ["...", "• • •"]:
                    return text
            except Exception as e:
                print(f"   ⚠️ inner_text failed: {e}")
            
            try:
                text = last_elem.text_content().strip()
                if text and text not in ["...", "• • •"]:
                    return text
            except Exception as e:
                print(f"   ⚠️ text_content failed: {e}")
            
            try:
                text = last_elem.evaluate("el => el.innerText || el.textContent").strip()
                if text and text not in ["...", "• • •"]:
                    return text
            except Exception as e:
                print(f"   ⚠️ evaluate failed: {e}")
            
            if attempt < retry_count - 1:
                print(f"   ⚠️ All extraction methods failed, retry {attempt + 1}/{retry_count}")
                time.sleep(1)
                
        except Exception as e:
            print(f"   ⚠️ Error extracting Mila message (attempt {attempt + 1}): {e}")
            if attempt < retry_count - 1:
                time.sleep(1)
    
    return None


def wait_for_new_message(page: Page, selector: str, previous_count: int, timeout_ms=30000):
    """Wait for a new message to appear"""
    end = time.time() + timeout_ms / 1000
    while time.time() < end:
        current = len(page.query_selector_all(selector))
        if current > previous_count:
            return current
        time.sleep(0.2)
    raise PlaywrightTimeoutError("No new message in time.")


def dismiss_cookie_banner(page: Page):
    """Dismiss cookie banners"""
    logger.debug("Attempting to dismiss cookie banner")
    
    for label in ["Accept all", "Reject non-essential services", "Customise settings"]:
        try:
            btn = page.get_by_role("button", name=label)
            if btn.is_visible():
                logger.info(f"➡️ Clicking cookie banner: {label}")
                print(f"➡️ Clicking cookie banner: {label}")
                btn.click()
                page.wait_for_timeout(800)
                return
        except Exception:
            pass

    try:
        btn = page.locator("button", has_text="Accept")
        if btn.first.is_visible():
            logger.info("➡️ Clicking generic Accept cookie button")
            print("➡️ Clicking generic Accept cookie button")
            btn.first.click()
            page.wait_for_timeout(800)
            return
    except Exception:
        pass

    logger.debug("ℹ️ Cookie banner not found or already dismissed")
    print("ℹ️ Cookie banner not found or already dismissed.")


def perform_mila_login(page: Page):
    """Perform Mila login"""
    logger.info("Attempting Mila login")
    
    if not MILA_LOGIN_USER or not MILA_LOGIN_PASS:
        logger.warning("⚠️ MILA_LOGIN_USER / MILA_LOGIN_PASS not set — skipping login")
        print("⚠️ MILA_LOGIN_USER / MILA_LOGIN_PASS not set — skipping login.")
        return

    try:
        logger.debug("🔐 Waiting for login form...")
        print("🔐 Waiting for login form...")

        username_locator = page.get_by_label(
            "Login by username/email address", exact=False
        )
        if username_locator.count() == 0:
            logger.debug("ℹ️ Username label not found, using generic input fallback")
            print("ℹ️ Username label not found, using generic input fallback.")
            username_locator = page.locator(
                "input:not([type='password']):not([type='hidden']):not([type='checkbox']):not([type='radio'])"
            ).first
        username_locator.wait_for(timeout=8000)

        password_locator = page.get_by_label("Password", exact=False)
        if password_locator.count() == 0:
            logger.debug("ℹ️ Password label not found, using input[type='password'] fallback")
            print("ℹ️ Password label not found, using input[type='password'] fallback.")
            password_locator = page.locator("input[type='password']").first
        password_locator.wait_for(timeout=8000)

        logger.info(f"➡️ Filling username: {MILA_LOGIN_USER}")
        print(f"➡️ Filling username: {MILA_LOGIN_USER}")
        username_locator.click()
        username_locator.fill(MILA_LOGIN_USER)

        logger.debug("➡️ Filling password")
        print("➡️ Filling password")
        password_locator.click()
        password_locator.fill(MILA_LOGIN_PASS)

        logger.debug("➡️ Clicking Log in button...")
        print("➡️ Clicking Log in button...")
        login_btn = page.get_by_role("button", name="Log in")
        login_btn.click()

        page.wait_for_timeout(2000)
        logger.info("🔓 Login attempt complete")
        print("🔓 Login attempt complete.")
        
    except PlaywrightTimeoutError:
        logger.info("ℹ️ Login form did not appear (maybe already logged in?)")
        print("ℹ️ Login form did not appear (maybe already logged in?).")
    except Exception as e:
        log_exception(logger, e, "perform_mila_login")
        print(f"❌ Error during login: {e}")


def open_mila_chat(page: Page):
    """Open Mila chat widget"""
    logger.debug("Attempting to open Mila chat widget")
    
    try:
        header = page.locator(SELECTORS["mila"]["header"]).first
        header.wait_for(state="visible", timeout=10000)
        aria_expanded = header.get_attribute("aria-expanded")
        if aria_expanded == "false":
            logger.info("💬 Opening Mila chat widget...")
            print("💬 Opening Mila chat widget...")
            header.click()
            page.wait_for_timeout(800)
        else:
            logger.debug("💬 Mila chat widget already open")
            print("💬 Mila chat widget already open.")
    except Exception as e:
        log_exception(logger, e, "open_mila_chat")
        print(f"ℹ️ Could not open Mila chat widget automatically: {e}")


def clear_mila_history(page: Page):
    """Clear Mila chat history"""
    logger.info("Clearing Mila chat history")
    
    try:
        print("🗑️  Clearing Mila chat history...")
        
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
        
        for menu_sel in menu_selectors:
            try:
                menu_btn = page.locator(menu_sel).first
                if menu_btn.is_visible(timeout=2000):
                    logger.debug(f"   Found menu button with selector: {menu_sel}")
                    print(f"   Found menu button with selector: {menu_sel}")
                    menu_btn.click()
                    page.wait_for_timeout(800)
                    menu_opened = True
                    break
            except Exception:
                continue
        
        if not menu_opened:
            logger.warning("   ⚠️ Could not find menu button, trying direct access...")
            print("   ⚠️ Could not find menu button, trying direct access...")
        
        clear_selectors = [
            "a.clear-history",
            "a[class*='clear-history']",
            "a.chat-dropdown-link:has-text('Clear History')",
            "a:has-text('Clear History')",
            "button:has-text('Clear History')",
        ]
        
        clicked = False
        for clear_sel in clear_selectors:
            try:
                clear_btn = page.locator(clear_sel).first
                if clear_btn.is_visible(timeout=2000):
                    logger.debug(f"   Found Clear History with selector: {clear_sel}")
                    print(f"   Found Clear History with selector: {clear_sel}")
                    clear_btn.click()
                    logger.info("   ✓ Clicked 'Clear History'")
                    print("   ✓ Clicked 'Clear History'")
                    page.wait_for_timeout(1500)
                    clicked = True
                    break
            except Exception:
                continue
        
        if not clicked:
            try:
                clear_link = page.get_by_text("Clear History", exact=False)
                if clear_link.is_visible(timeout=2000):
                    clear_link.click()
                    logger.info("   ✓ Clicked 'Clear History' using text match")
                    print("   ✓ Clicked 'Clear History' using text match")
                    page.wait_for_timeout(1500)
                    clicked = True
            except Exception:
                pass
        
        if not clicked:
            logger.warning("   ⚠️ Could not find 'Clear History' button")
            print("   ⚠️ Could not find 'Clear History' button")
            return
        
        try:
            confirm_selectors = [
                "button:has-text('Confirm')",
                "button:has-text('Yes')",
                "button:has-text('OK')",
                "button:has-text('Clear')",
            ]
            
            for confirm_sel in confirm_selectors:
                try:
                    confirm_btn = page.locator(confirm_sel).first
                    if confirm_btn.is_visible(timeout=2000):
                        confirm_btn.click()
                        logger.info("   ✓ Confirmed history clearing")
                        print("   ✓ Confirmed history clearing")
                        page.wait_for_timeout(1000)
                        break
                except Exception:
                    continue
        except Exception:
            pass
        
        logger.info("   ✓ History cleared successfully")
        print("   ✓ History cleared successfully")
        
    except Exception as e:
        log_exception(logger, e, "clear_mila_history")
        print(f"   ❌ Error clearing Mila history: {e}")


def ensure_mila_chat_open(page: Page):
    """Ensure Mila chat is open"""
    try:
        header = page.locator(SELECTORS["mila"]["header"]).first
        if header.is_visible():
            aria_expanded = header.get_attribute("aria-expanded")
            if aria_expanded == "false":
                print("💬 Reopening Mila chat widget...")
                header.click()
                page.wait_for_timeout(800)
        
        page.evaluate("""
            const chatContainer = document.querySelector('div.ai-deepchat--body, div[class*="chat-messages"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        """)
        page.wait_for_timeout(500)
    except Exception as e:
        print(f"⚠️ Could not ensure chat is open: {e}")


def find_mila_input(page: Page) -> Optional[any]:
    """Find Mila input field"""
    for selector in SELECTORS["mila"]["input_selectors"]:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                try:
                    locator.wait_for(state="attached", timeout=2000)
                    print(f"✓ Found Mila input with selector: {selector}")
                    return locator
                except Exception:
                    continue
        except Exception:
            continue
    return None


def send_message_to_mila(page: Page, text: str, max_retries: int = 3):
    """Send message to Mila"""
    send_sel = SELECTORS["mila"]["send_button"]

    for attempt in range(max_retries):
        try:
            ensure_mila_chat_open(page)
            
            box = find_mila_input(page)
            if box is None:
                raise Exception("Could not find Mila input field")
            
            box.wait_for(state="visible", timeout=30000)
            box.click()
            page.wait_for_timeout(300)
            
            try:
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
            except Exception as e:
                print(f"   ⚠️ JavaScript fill failed, trying keyboard: {e}")
                box.fill("")
                page.keyboard.type(text, delay=0)

            btn = page.locator(send_sel).first
            try:
                btn.wait_for(state="visible", timeout=5000)
                btn.click()
            except PlaywrightTimeoutError:
                print("⚠️ Send button not visible; pressing Enter")
                try:
                    page.keyboard.press("Enter")
                except Exception as e:
                    print(f"⚠️ Failed to press Enter: {e}")
            
            print("✓ Message sent to Mila")
            return
            
        except PlaywrightTimeoutError as e:
            if attempt == max_retries - 1:
                print(f"❌ Failed after {max_retries} attempts")
                raise
            print(f"⚠️ Retry {attempt + 1}/{max_retries}...")
            page.wait_for_timeout(5000)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"⚠️ Retry {attempt + 1}/{max_retries} - Error: {e}")
            page.wait_for_timeout(5000)


# =====================================
# MAIN BRIDGE FUNCTION
# =====================================

def run_bridge() -> Tuple[List[Dict[str, str]], int]:
    """
    Run the headless bridge.
    Returns: (conversation_log, number_of_turns)
    
    NEW FEATURES:
    - Tracks actual number of turns completed
    - Exits early if completeness score >= 85
    """
    global CONVERSATION_LOG
    CONVERSATION_LOG = []
    
    # Track actual turns completed
    turns_completed = 0

    logger.info("=" * 80)
    logger.info("Starting headless bridge bot")
    logger.info("=" * 80)
    logger.info(f"Mila URL:        {MILA_URL}")
    logger.info(f"Tester API URL:  {TESTER_API_URL}")
    logger.info(f"Max turns:       {MAX_TURNS}")
    logger.info("=" * 80)
    
    print("Starting headless bridge bot...")
    print(f"Mila URL:        {MILA_URL}")
    print(f"Tester API URL:  {TESTER_API_URL}")
    print(f"Max turns:       {MAX_TURNS}")

    if not MILA_URL:
        logger.error("❌ Missing MILA_URL in .env")
        print("❌ Missing MILA_URL in .env")
        return CONVERSATION_LOG, turns_completed

    # Create tester session
    if not create_tester_session():
        logger.error("❌ Failed to create tester session")
        print("❌ Failed to create tester session")
        return CONVERSATION_LOG, turns_completed

    logger.info("Launching browser in headless mode")
    with sync_playwright() as p:
        # Launch browser in HEADLESS mode
        logger.debug("Browser args: headless=True, --no-sandbox, --disable-dev-shm-usage")
        browser = p.chromium.launch(
            headless=True,  # ← Headless!
            args=['--no-sandbox', '--disable-dev-shm-usage']  # For Linux/Docker
        )
        logger.info("✓ Browser launched successfully")

        ctx_kwargs_mila = {
            "viewport": {"width": 1280, "height": 720},
            "ignore_https_errors": True,
        }
        if MILA_HTTP_USER and MILA_HTTP_PASS:
            ctx_kwargs_mila["http_credentials"] = {
                "username": MILA_HTTP_USER,
                "password": MILA_HTTP_PASS,
            }

        context_mila = browser.new_context(**ctx_kwargs_mila)
        page_mila = context_mila.new_page()

        # ----- Mila -----
        logger.info(f"➡️ Opening Mila at {MILA_URL}")
        print("➡️ Opening Mila...")
        page_mila.goto(MILA_URL)
        page_mila.wait_for_load_state("networkidle")
        logger.debug("✓ Page loaded (networkidle state)")

        dismiss_cookie_banner(page_mila)
        perform_mila_login(page_mila)
        page_mila.wait_for_timeout(2000)
        open_mila_chat(page_mila)
        clear_mila_history(page_mila)

        mila_sel = SELECTORS["mila"]["message_bubbles"]

        # ===== FIRST MILA MESSAGE =====
        initial_mila_count = len(page_mila.query_selector_all(mila_sel))
        logger.info(f"⏳ Waiting for Mila's first message... (current bubbles: {initial_mila_count})")
        print(f"⏳ Waiting for Mila's first message... (bubbles: {initial_mila_count})")
        try:
            wait_for_new_message(page_mila, mila_sel, previous_count=0, timeout_ms=30000)
            wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=30000)
        except PlaywrightTimeoutError:
            logger.error("❌ Mila didn't send first message (timeout)")
            print("❌ Mila didn't send first message")
            browser.close()
            return CONVERSATION_LOG, turns_completed

        mila_count = len(page_mila.query_selector_all(mila_sel))
        mila_last = get_mila_last_message_text(page_mila, mila_sel)
        
        if not mila_last:
            logger.error("❌ Failed to extract Mila's first message")
            print("❌ Failed to extract Mila's first message")
            browser.close()
            return CONVERSATION_LOG, turns_completed
        
        logger.info(f"🟠 Mila first message ({len(mila_last)} chars):")
        logger.debug(f"   {mila_last}")
        print(f"🟠 Mila first message:\n{mila_last}\n")
        CONVERSATION_LOG.append({"speaker": "Mila", "message": mila_last})

        # ===== MAIN LOOP =====
        logger.info(f"Starting main conversation loop (max {MAX_TURNS} turns)")
        early_exit = False
        early_exit_reason = ""
        
        for turn in range(1, MAX_TURNS + 1):
            logger.info("=" * 60)
            logger.info(f"TURN {turn}/{MAX_TURNS}")
            logger.info("=" * 60)
            print(f"========== TURN {turn} ==========")

            # Mila → Tester (via HTTP API)
            logger.debug(f"Sending Mila's message to tester API ({len(mila_last)} chars)")
            result = send_to_tester_api(mila_last)
            if result is None:
                logger.error("❌ Tester API failed, stopping conversation")
                print("❌ Tester API failed")
                break
            
            tester_reply, score, should_continue = result
            
            # Format reply with score if available
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
                logger.info(f"🧪 Tester reply with score {score}/100, should_continue={should_continue}")
                
                # ✅ CHECK FOR EARLY EXIT CONDITIONS
                # Condition 1: High score (>= 85)
                if score >= 85:
                    logger.info(f"🎯 HIGH SCORE DETECTED: {score}/100 >= 85")
                    early_exit = True
                    early_exit_reason = f"High score ({score}/100)"
                
                # ✅ Condition 2: should_continue=false AND score >= 80
                if not should_continue and score >= 80:
                    logger.info(f"✅ COMPLETION SIGNAL: should_continue=false AND score={score}/100 >= 80")
                    early_exit = True
                    early_exit_reason = f"Completion signal (score={score}/100, should_continue=false)"
                
                if early_exit:
                    logger.info(f"✨ Exiting early - {early_exit_reason}")
                    print(f"\n✨ EARLY EXIT: {early_exit_reason}")
                    print(f"✅ Test case completed successfully!\n")
            else:
                display_reply = tester_reply
                logger.info(f"🧪 Tester reply without score, should_continue={should_continue}")
            
            logger.debug(f"   Reply: {tester_reply[:100]}...")
            print(f"🧪 Tester reply:\n{display_reply}\n")
            CONVERSATION_LOG.append({"speaker": "Tester", "message": display_reply})
            
            # Increment turn counter (one full turn = Mila → Tester → Mila)
            turns_completed = turn
            
            # Exit early if conditions met
            if early_exit:
                logger.info(f"✅ Completed {turns_completed} turns (early exit: {early_exit_reason})")
                print(f"✅ Completed {turns_completed} turns (early exit: {early_exit_reason})")
                break

            # Tester → Mila
            logger.debug(f"Sending tester's reply to Mila ({len(tester_reply)} chars)")
            try:
                send_message_to_mila(page_mila, tester_reply)
            except Exception as e:
                log_exception(logger, e, "send_message_to_mila")
                logger.error(f"❌ Failed to send to Mila: {e}")
                print(f"❌ Failed to send to Mila: {e}")
                break

            logger.debug("Waiting for Mila's response...")
            try:
                new_mila_count = wait_for_new_message(
                    page_mila, mila_sel, mila_count, timeout_ms=45000
                )
                wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=60000)
            except PlaywrightTimeoutError:
                logger.error("❌ Mila did not reply (timeout)")
                print("❌ Mila did not reply")
                break

            mila_count = new_mila_count
            mila_last = get_mila_last_message_text(page_mila, mila_sel)
            
            if not mila_last:
                logger.error("❌ Failed to extract Mila's reply")
                print("❌ Failed to extract Mila's reply")
                break
            
            logger.info(f"🟠 Mila reply ({len(mila_last)} chars):")
            logger.debug(f"   {mila_last[:100]}...")
            print(f"🟠 Mila reply:\n{mila_last}\n")
            CONVERSATION_LOG.append({"speaker": "Mila", "message": mila_last})

        logger.info("=" * 80)
        logger.info(f"✅ Chat loop finished! Total turns: {turns_completed}")
        logger.info(f"   Conversation entries: {len(CONVERSATION_LOG)}")
        logger.info("=" * 80)
        print(f"✅ Chat loop finished! Completed {turns_completed} turns")
        time.sleep(2)
        
        logger.debug("Closing browser")
        browser.close()
        logger.info("✓ Browser closed")

    logger.info(f"Returning conversation log with {len(CONVERSATION_LOG)} entries and {turns_completed} turns")
    return CONVERSATION_LOG, turns_completed


if __name__ == "__main__":
    logger.info("Script started from command line")
    logger.info(f"Command line arguments: {sys.argv}")
    
    try:
        conversation_log, num_turns = run_bridge()
        logger.info(f"✅ Script completed successfully")
        logger.info(f"   Conversation entries: {len(conversation_log)}")
        logger.info(f"   Number of turns: {num_turns}")
        print(f"\n📊 Summary:")
        print(f"   Conversation entries: {len(conversation_log)}")
        print(f"   Number of turns: {num_turns}")
    except KeyboardInterrupt:
        logger.warning("⚠️ Script interrupted by user (Ctrl+C)")
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        log_exception(logger, e, "main script execution")
        logger.error(f"❌ Script failed: {e}")
        raise
    finally:
        logger.info("Script execution finished")