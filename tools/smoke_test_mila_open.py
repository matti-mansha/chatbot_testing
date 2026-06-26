"""
Targeted smoke test: verify the FAB/state fix actually opens MILA's chat.

Reuses playwright_bridge_bot_headless.py's own functions so we test the same
code paths run_bridge_with_restart would hit in production. Logs in, opens
the chat, asserts the container reaches chat-open state, asserts #text-input
is visible. Exits 0 on success, 1 on any failure with a concrete reason.

Run directly on the server:
    env/bin/python smoke_test_mila_open.py
"""
import sys
import traceback

from playwright.sync_api import sync_playwright

# Reuse the (now fixed) helpers from the real bridge bot module
from playwright_bridge_bot_headless import (
    MILA_URL,
    MILA_HTTP_USER,
    MILA_HTTP_PASS,
    SELECTORS,
    dismiss_cookie_banner,
    perform_mila_login,
    open_mila_chat,
    is_mila_chat_open,
    get_mila_chat_container_classes,
    ensure_mila_chat_open,
)


def fail(msg: str) -> None:
    print(f"\n❌ SMOKE TEST FAILED: {msg}\n")
    sys.exit(1)


def main() -> None:
    if not MILA_URL:
        fail("MILA_URL not set in .env")

    print(f"▶ Launching headless chromium")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        ctx_kwargs = {"viewport": {"width": 1280, "height": 720}, "ignore_https_errors": True}
        if MILA_HTTP_USER and MILA_HTTP_PASS:
            ctx_kwargs["http_credentials"] = {
                "username": MILA_HTTP_USER,
                "password": MILA_HTTP_PASS,
            }
        ctx = browser.new_context(**ctx_kwargs)
        page = ctx.new_page()

        print(f"▶ Navigating to {MILA_URL}")
        page.goto(MILA_URL)
        page.wait_for_load_state("networkidle")

        dismiss_cookie_banner(page)
        perform_mila_login(page)
        page.wait_for_timeout(2000)

        print("\n--- STATE BEFORE open_mila_chat ---")
        classes_before = get_mila_chat_container_classes(page)
        print(f"  chat-container classes: {classes_before!r}")
        print(f"  is_mila_chat_open: {is_mila_chat_open(page)}")

        print("\n▶ Calling open_mila_chat()")
        try:
            open_mila_chat(page)
        except Exception as e:
            fail(f"open_mila_chat raised: {type(e).__name__}: {e}")

        print("\n--- STATE AFTER open_mila_chat ---")
        classes_after = get_mila_chat_container_classes(page)
        print(f"  chat-container classes: {classes_after!r}")
        is_open = is_mila_chat_open(page)
        print(f"  is_mila_chat_open: {is_open}")

        if not is_open:
            fail(
                f"Chat container did not reach chat-open state. "
                f"classes='{classes_after}'"
            )

        print("\n▶ Checking #text-input visibility")
        input_sel = "#text-input[contenteditable='true']"
        box = page.locator(input_sel).first
        try:
            box.wait_for(state="visible", timeout=10000)
        except Exception as e:
            fail(
                f"#text-input is still not visible after open. "
                f"classes='{classes_after}' err={e}"
            )
        print("  ✓ #text-input is visible")

        print("\n▶ Calling ensure_mila_chat_open() (should be a no-op)")
        try:
            ensure_mila_chat_open(page)
        except Exception as e:
            fail(f"ensure_mila_chat_open raised: {type(e).__name__}: {e}")
        print(f"  ✓ ensure_mila_chat_open survived")
        print(f"  final classes: {get_mila_chat_container_classes(page)!r}")

        browser.close()

    print("\n" + "=" * 60)
    print("✅ SMOKE TEST PASSED — chat opens and input is visible")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("\n❌ UNCAUGHT EXCEPTION:")
        traceback.print_exc()
        sys.exit(2)
