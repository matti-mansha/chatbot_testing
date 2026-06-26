# playwright_bridge_bot.py
import os
import time
import pathlib
from typing import Optional, List, Dict

from playwright.sync_api import (
    sync_playwright,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)
from dotenv import load_dotenv

# =====================================
# LOAD .env
# =====================================

BASE_DIR = pathlib.Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

MILA_URL = os.getenv("MILA_URL", "").strip()
TEST_BOT_URL = os.getenv("TEST_BOT_URL", "").strip()
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

# Mila web login credentials
MILA_LOGIN_USER = os.getenv("MILA_LOGIN_USER", "").strip()
MILA_LOGIN_PASS = os.getenv("MILA_LOGIN_PASS", "").strip()

# Optional HTTP basic auth (for the whole stage site)
MILA_HTTP_USER = os.getenv("MILA_HTTP_USER")
MILA_HTTP_PASS = os.getenv("MILA_HTTP_PASS")

# Tester placeholders (for Streamlit sidebar)
# These are passed as command-line arguments from run_test_executions.py
# Format: python playwright_bridge_bot.py <test_case> <persona> <test_case_details> <test_case_prompt>
import sys

TEST_CASE = sys.argv[1] if len(sys.argv) > 1 else "Onboarding & \"How it works\""
TEST_PERSONA = sys.argv[2] if len(sys.argv) > 2 else "Host family in Spain"
TEST_CASE_DETAILS = sys.argv[3] if len(sys.argv) > 3 else ""
TEST_CASE_PROMPT = sys.argv[4] if len(sys.argv) > 4 else ""

# Conversation log (used by run_test_executions.py to dump into Notion)
CONVERSATION_LOG: List[Dict[str, str]] = []


# =====================================
# SELECTORS
# =====================================

SELECTORS = {
    "mila": {
        # Mila assistant bubble: <div class="message-bubble ai-message ai-message-text">...</div>
        "message_bubbles": ".message-bubble.ai-message.ai-message-text",
        # Multiple fallback selectors for input - based on screenshot showing "Enter a prompt here"
        "input_selectors": [
            "input[placeholder='Enter a prompt here']",
            "textarea[placeholder='Enter a prompt here']",
            "[contenteditable='true']",
            "div.ai-deepchat--input-container input",
            "div.ai-deepchat--input-container textarea",
            "textarea.ai-deepchat--textarea",
            "input[type='text']:not([name='q'])",  # Exclude search box
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
    },
    "tester": {
        "message_bubbles": "div[data-testid='stChatMessage']",
        # Streamlit chat_input placeholder
        "input": "textarea[placeholder='Type your message...']",
    },
}


# =====================================
# SCREEN SIZE + WINDOW ARRANGE
# =====================================

def get_screen_size():
    """Return (width, height); fall back if Tk not available."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1600, 900


def arrange_windows_side_by_side(page_left: Page, page_right: Page):
    """Best-effort: Mila left, Tester right."""
    screen_w, screen_h = get_screen_size()
    half_w = int(screen_w // 2)
    height = int(screen_h * 0.9)

    page_left.set_viewport_size({"width": half_w, "height": height})
    page_right.set_viewport_size({"width": half_w, "height": height})

    try:
        # Mila
        ctx_left = page_left.context
        session_left = ctx_left.new_cdp_session(page_left)
        win_left = session_left.send("Browser.getWindowForTarget")
        session_left.send(
            "Browser.setWindowBounds",
            {
                "windowId": win_left["windowId"],
                "bounds": {"left": 0, "top": 0, "width": half_w, "height": height},
            },
        )

        # Tester
        ctx_right = page_right.context
        session_right = ctx_right.new_cdp_session(page_right)
        win_right = session_right.send("Browser.getWindowForTarget")
        session_right.send(
            "Browser.setWindowBounds",
            {
                "windowId": win_right["windowId"],
                "bounds": {
                    "left": half_w,
                    "top": 0,
                    "width": half_w,
                    "height": height,
                },
            },
        )
    except Exception as e:
        print(f"ℹ️ Could not fully position windows via CDP (viewport ok): {e}")


# =====================================
# GENERIC HELPERS
# =====================================

def wait_for_mila_typing_complete(page: Page, selector: str, timeout_ms: int = 60000):
    """
    Wait for Mila to finish typing. Mila shows "..." while generating response.
    We wait until the text content stops changing and is not just "...".
    """
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
                
                # Check if it's the loading indicator
                if current_text in ("...", "• • •", ""):
                    last_text = None
                    stable_count = 0
                    time.sleep(0.5)
                    continue
                
                # Check if text has stopped changing
                if current_text == last_text:
                    stable_count += 1
                    if stable_count >= 2:
                        print("✓ Mila finished typing")
                        return
                else:
                    stable_count = 0
                    last_text = current_text
                
            time.sleep(0.5)
        except Exception as e:
            print(f"   ⚠️ Error checking typing status: {e}")
            time.sleep(0.5)
    
    print("⚠️ Typing wait timeout, proceeding anyway...")


def get_mila_last_message_text(page: Page, selector: str, retry_count: int = 3) -> Optional[str]:
    """Get the last message text from Mila chatbot with retries."""
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
            
            # inner_text
            try:
                text = last_elem.inner_text().strip()
                if text and text not in ["...", "• • •"]:
                    return text
            except Exception as e:
                print(f"   ⚠️ inner_text failed: {e}")
            
            # text_content
            try:
                text = last_elem.text_content().strip()
                if text and text not in ["...", "• • •"]:
                    return text
            except Exception as e:
                print(f"   ⚠️ text_content failed: {e}")
            
            # evaluate
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


def get_tester_last_message_text(page: Page, selector: str) -> Optional[str]:
    """Get the last message text from Tester (Streamlit), including completeness score badges."""
    elems = page.query_selector_all(selector)
    if not elems:
        return None
    
    last_elem = elems[-1]
    
    # Get ALL text from the chat message container (including score badges)
    # Don't just get the first markdown element - get everything!
    full_text = last_elem.inner_text().strip()
    
    # Filter out icon names at the start
    icon_names = [
        "face", "smart_toy", "assistant", "user", "person", "robot",
        "account_circle", "support_agent", "android", "psychology", "chat",
    ]
    
    lines = full_text.split("\n")
    if lines and lines[0].strip().lower() in icon_names:
        return "\n".join(lines[1:]).strip()
    
    words = full_text.split(None, 1)
    if len(words) > 1 and words[0].strip().lower() in icon_names:
        return words[1].strip()
    
    return full_text


def wait_for_new_message(page: Page, selector: str, previous_count: int, timeout_ms=30000):
    """Wait for a new message to appear."""
    end = time.time() + timeout_ms / 1000
    while time.time() < end:
        current = len(page.query_selector_all(selector))
        if current > previous_count:
            return current
        time.sleep(0.2)
    raise PlaywrightTimeoutError("No new message in time.")


def dismiss_cookie_banner(page: Page):
    for label in ["Accept all", "Reject non-essential services", "Customise settings"]:
        try:
            btn = page.get_by_role("button", name=label)
            if btn.is_visible():
                print(f"➡️ Clicking cookie banner: {label}")
                btn.click()
                page.wait_for_timeout(800)
                return
        except Exception:
            pass

    try:
        btn = page.locator("button", has_text="Accept")
        if btn.first.is_visible():
            print("➡️ Clicking generic Accept cookie button")
            btn.first.click()
            page.wait_for_timeout(800)
            return
    except Exception:
        pass

    print("ℹ️ Cookie banner not found or already dismissed.")


# =====================================
# MILA HELPERS
# =====================================

def perform_mila_login(page: Page):
    if not MILA_LOGIN_USER or not MILA_LOGIN_PASS:
        print("⚠️ MILA_LOGIN_USER / MILA_LOGIN_PASS not set — skipping login.")
        return

    try:
        print("🔐 Waiting for login form...")

        username_locator = page.get_by_label(
            "Login by username/email address", exact=False
        )
        if username_locator.count() == 0:
            print("ℹ️ Username label not found, using generic input fallback.")
            username_locator = page.locator(
                "input:not([type='password']):not([type='hidden']):not([type='checkbox']):not([type='radio'])"
            ).first
        username_locator.wait_for(timeout=8000)

        password_locator = page.get_by_label("Password", exact=False)
        if password_locator.count() == 0:
            print("ℹ️ Password label not found, using input[type='password'] fallback.")
            password_locator = page.locator("input[type='password']").first
        password_locator.wait_for(timeout=8000)

        print(f"➡️ Filling username: {MILA_LOGIN_USER}")
        username_locator.click()
        username_locator.fill(MILA_LOGIN_USER)

        print("➡️ Filling password")
        password_locator.click()
        password_locator.fill(MILA_LOGIN_PASS)

        print("➡️ Clicking Log in button...")
        login_btn = page.get_by_role("button", name="Log in")
        login_btn.click()

        page.wait_for_timeout(2000)
        print("🔓 Login attempt complete.")
    except PlaywrightTimeoutError:
        print("ℹ️ Login form did not appear (maybe already logged in?).")
    except Exception as e:
        print(f"❌ Error during login: {e}")


def open_mila_chat(page: Page):
    try:
        header = page.locator(SELECTORS["mila"]["header"]).first
        header.wait_for(state="visible", timeout=10000)
        aria_expanded = header.get_attribute("aria-expanded")
        if aria_expanded == "false":
            print("💬 Opening Mila chat widget...")
            header.click()
            page.wait_for_timeout(800)
        else:
            print("💬 Mila chat widget already open.")
    except Exception as e:
        print(f"ℹ️ Could not open Mila chat widget automatically: {e}")


def clear_mila_history(page: Page):
    """Clear Mila chat history before starting the test."""
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
                    print(f"   Found menu button with selector: {menu_sel}")
                    menu_btn.click()
                    page.wait_for_timeout(800)
                    menu_opened = True
                    break
            except Exception:
                continue
        
        if not menu_opened:
            print("   ⚠️ Could not find menu button, trying direct access to Clear History...")
        
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
                    print(f"   Found Clear History with selector: {clear_sel}")
                    clear_btn.click()
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
                    print("   ✓ Clicked 'Clear History' using text match")
                    page.wait_for_timeout(1500)
                    clicked = True
            except Exception:
                pass
        
        if not clicked:
            print("   ⚠️ Could not find 'Clear History' button")
            page.screenshot(path="mila_clear_history_debug.png")
            print("   📸 Screenshot saved: mila_clear_history_debug.png")
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
                        print("   ✓ Confirmed history clearing")
                        page.wait_for_timeout(1000)
                        break
                except Exception:
                    continue
        except Exception:
            pass
        
        print("   ✓ History cleared successfully")
        
    except Exception as e:
        print(f"   ❌ Error clearing Mila history: {e}")
        page.screenshot(path="mila_clear_history_error.png")
        print("   📸 Screenshot saved: mila_clear_history_error.png")


def ensure_mila_chat_open(page: Page):
    """Ensure the Mila chat widget is open and scrolled to bottom."""
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
    """Try multiple selectors to find the Mila input field."""
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
    """Send message to Mila with retry logic."""
    send_sel = SELECTORS["mila"]["send_button"]

    for attempt in range(max_retries):
        try:
            ensure_mila_chat_open(page)
            
            box = find_mila_input(page)
            if box is None:
                raise Exception("Could not find Mila input field with any selector")
            
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
                print(f"   ⚠️ JavaScript fill failed, trying keyboard input: {e}")
                box.fill("")
                page.keyboard.type(text, delay=0)

            btn = page.locator(send_sel).first
            try:
                btn.wait_for(state="visible", timeout=5000)
                btn.click()
            except PlaywrightTimeoutError:
                print("⚠️ Mila send button not visible; pressing Enter in input.")
                try:
                    page.keyboard.press("Enter")
                except Exception as e:
                    print(f"⚠️ Mila: failed to press Enter: {e}")
            
            print("✓ Message sent to Mila")
            return
            
        except PlaywrightTimeoutError as e:
            if attempt == max_retries - 1:
                print(f"❌ Failed to send message to Mila after {max_retries} attempts")
                print(f"   Last error: {e}")
                raise
            print(f"⚠️ Retry {attempt + 1}/{max_retries} for Mila input...")
            page.wait_for_timeout(5000)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Unexpected error sending to Mila: {e}")
                raise
            print(f"⚠️ Retry {attempt + 1}/{max_retries} - Error: {e}")
            page.wait_for_timeout(5000)


# =====================================
# TESTER (STREAMLIT) HELPERS
# =====================================

def fill_tester_placeholders(page: Page):
    """Fills 'Test case', 'Persona', 'Test case details', and 'Test Case Prompt' in the Streamlit sidebar."""
    try:
        print("📝 Filling tester fields in sidebar...")
        
        # Fill Test case
        tc = page.get_by_label("Test case ({{test_case}})", exact=False)
        tc.wait_for(timeout=10000)
        tc.fill(TEST_CASE)
        print(f"   ✓ Filled Test case: {TEST_CASE[:50]}...")

        # Fill Persona
        persona = page.get_by_label("Persona ({{persona}})", exact=False)
        persona.wait_for(timeout=10000)
        persona.fill(TEST_PERSONA)
        print(f"   ✓ Filled Persona: {TEST_PERSONA[:50]}...")

        # Fill Test case details (optional)
        try:
            details = page.get_by_label("Test case details ({{test_case_details}})", exact=False)
            details.wait_for(timeout=10000)
            details.fill(TEST_CASE_DETAILS or "")
            if TEST_CASE_DETAILS:
                print(f"   ✓ Filled Test case details: {TEST_CASE_DETAILS[:50]}...")
        except Exception:
            # Older app versions may not have this field; ignore
            pass

        # Fill Test Case Prompt (override) - THIS IS KEY!
        try:
            prompt_field = page.get_by_label("Test Case Prompt (override)", exact=False)
            prompt_field.wait_for(timeout=10000)
            prompt_field.fill(TEST_CASE_PROMPT or "")
            if TEST_CASE_PROMPT:
                print(f"   ✓ Filled Test Case Prompt: {len(TEST_CASE_PROMPT)} chars")
            else:
                print(f"   ⚠️ Test Case Prompt is empty")
        except Exception as e:
            print(f"   ⚠️ Could not fill Test Case Prompt field: {e}")

        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"⚠️ Could not fill tester placeholders: {e}")


def send_message_to_tester(page: Page, text: str):
    """Send message to Streamlit chat input."""
    if not text or text == "None":
        print("❌ Cannot send empty or None message to Tester")
        raise ValueError("Message text is empty or None")
    
    input_sel = SELECTORS["tester"]["input"]
    box = page.locator(input_sel).first
    box.wait_for(state="visible", timeout=10000)
    box.click()
    page.wait_for_timeout(300)
    
    box.fill("")
    box.fill(text)
    page.wait_for_timeout(500)

    try:
        box.press("Enter")
    except Exception as e:
        print(f"⚠️ Tester: failed to press Enter: {e}")
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass


def wait_for_tester_response(page: Page, previous_count: int, timeout_ms: int = 120000) -> int:
    """Wait for Tester bot to respond."""
    tester_sel = SELECTORS["tester"]["message_bubbles"]
    
    print("⏳ Waiting for Tester bot (OpenAI call may take 30-90s)...")
    
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        print("⚠️ Network still active, continuing to check for messages...")
    
    page.wait_for_timeout(2000)
    
    end_time = time.time() + (timeout_ms / 1000)
    while time.time() < end_time:
        current = len(page.query_selector_all(tester_sel))
        if current > previous_count:
            page.wait_for_timeout(1000)
            return current
        time.sleep(0.5)
    
    raise PlaywrightTimeoutError(f"Tester did not reply within {timeout_ms}ms")


# =====================================
# MAIN BRIDGE FUNCTION
# =====================================

def run_bridge() -> List[Dict[str, str]]:
    """
    Run the bridge and return the full conversation as a list of:
    { "speaker": "Mila" | "Tester", "message": "<text>" }
    """
    global CONVERSATION_LOG
    CONVERSATION_LOG = []

    print("Starting bridge bot...")
    print(f"Mila URL:   {MILA_URL}")
    print(f"Tester URL: {TEST_BOT_URL}")
    print(f"Max turns:  {MAX_TURNS}")

    if not MILA_URL or not TEST_BOT_URL:
        print("❌ Missing MILA_URL or TEST_BOT_URL in .env")
        return CONVERSATION_LOG

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=100)

        screen_w, screen_h = get_screen_size()
        half_w = max(800, screen_w // 2)
        height = int(screen_h * 0.9)

        ctx_kwargs_mila = {
            "viewport": {"width": half_w, "height": height},
            "ignore_https_errors": True,
        }
        if MILA_HTTP_USER and MILA_HTTP_PASS:
            ctx_kwargs_mila["http_credentials"] = {
                "username": MILA_HTTP_USER,
                "password": MILA_HTTP_PASS,
            }

        context_mila = browser.new_context(**ctx_kwargs_mila)
        page_mila = context_mila.new_page()

        context_tester = browser.new_context(
            viewport={"width": half_w, "height": height},
            ignore_https_errors=True,
        )
        page_tester = context_tester.new_page()

        # ----- Mila -----
        print("➡️ Opening Mila...")
        page_mila.goto(MILA_URL)
        page_mila.wait_for_load_state("networkidle")
        page_mila.bring_to_front()

        dismiss_cookie_banner(page_mila)
        perform_mila_login(page_mila)
        page_mila.wait_for_timeout(2000)
        open_mila_chat(page_mila)
        
        clear_mila_history(page_mila)

        # ----- Tester -----
        print("➡️ Opening Tester bot...")
        page_tester.goto(TEST_BOT_URL)
        page_tester.wait_for_load_state("networkidle")
        fill_tester_placeholders(page_tester)

        arrange_windows_side_by_side(page_mila, page_tester)

        mila_sel = SELECTORS["mila"]["message_bubbles"]
        tester_sel = SELECTORS["tester"]["message_bubbles"]

        # ===== FIRST MILA MESSAGE =====
        initial_mila_count = len(page_mila.query_selector_all(mila_sel))
        print(
            f"⏳ Waiting for Mila's first chatbot message... "
            f"(initial bubbles found: {initial_mila_count})"
        )
        try:
            wait_for_new_message(page_mila, mila_sel, previous_count=0, timeout_ms=30000)
            wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=30000)
        except PlaywrightTimeoutError:
            print("❌ Timeout: Mila didn't send the first message.")
            browser.close()
            return CONVERSATION_LOG

        mila_count = len(page_mila.query_selector_all(mila_sel))
        mila_last = get_mila_last_message_text(page_mila, mila_sel)
        
        if not mila_last:
            print("❌ Failed to extract Mila's first message")
            browser.close()
            return CONVERSATION_LOG
            
        print(f"🟠 Mila first message:\n{mila_last}\n")
        CONVERSATION_LOG.append({"speaker": "Mila", "message": mila_last})

        tester_count = len(page_tester.query_selector_all(tester_sel))

        # ===== MAIN LOOP =====
        for turn in range(1, MAX_TURNS + 1):
            print(f"========== TURN {turn} ==========")

            # Mila → Tester
            try:
                send_message_to_tester(page_tester, mila_last)
            except ValueError as e:
                print(f"❌ {e}")
                break
            
            page_tester.wait_for_timeout(2000)
            current_tester_count = len(page_tester.query_selector_all(tester_sel))

            try:
                new_tester_count = wait_for_tester_response(
                    page_tester, current_tester_count, timeout_ms=120000
                )
            except PlaywrightTimeoutError:
                print("❌ Tester did not reply in time (OpenAI may have timed out).")
                break

            tester_count = new_tester_count
            tester_reply = get_tester_last_message_text(page_tester, tester_sel)
            
            if not tester_reply:
                print("❌ Failed to extract Tester's reply")
                break
                
            print(f"🧪 Tester reply:\n{tester_reply}\n")
            CONVERSATION_LOG.append({"speaker": "Tester", "message": tester_reply})

            # Tester → Mila
            try:
                send_message_to_mila(page_mila, tester_reply)
            except Exception as e:
                print(f"❌ Failed to send message to Mila: {e}")
                page_mila.screenshot(path="mila_error.png")
                print("   📸 Screenshot saved: mila_error.png")
                break

            try:
                new_mila_count = wait_for_new_message(
                    page_mila, mila_sel, mila_count, timeout_ms=45000
                )
                wait_for_mila_typing_complete(page_mila, mila_sel, timeout_ms=60000)
            except PlaywrightTimeoutError:
                print("❌ Mila did not reply in time.")
                break

            mila_count = new_mila_count
            mila_last = get_mila_last_message_text(page_mila, mila_sel)
            
            if not mila_last:
                print("❌ Failed to extract Mila's reply")
                page_mila.screenshot(path="mila_empty_response.png")
                print("   📸 Screenshot saved: mila_empty_response.png")
                break
                
            print(f"🟠 Mila reply:\n{mila_last}\n")
            CONVERSATION_LOG.append({"speaker": "Mila", "message": mila_last})

        print("✅ Chat loop finished!")
        time.sleep(3)
        browser.close()

    return CONVERSATION_LOG


if __name__ == "__main__":
    run_bridge()