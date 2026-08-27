"""
Reproduction test for the two P1 bridge fixes (login verification + notice overlay).

Run:  ./env/bin/python tools/test_bridge_overlay_fixes.py

Requires only playwright + chromium — no network, no Notion, no OpenAI, and it
never touches the live MILA site.

Builds a page that reproduces the exact production failure: a MILA chat FAB
covered by a persistent Drupal <qz-message> error banner.
"""
import ast, pathlib, sys
from playwright.sync_api import sync_playwright, Error as PWError

SRC = pathlib.Path(__file__).resolve().parent.parent / "playwright_bridge_bot_headless.py"

# Pull the real constants out of the source so this tests the shipped values.
tree = ast.parse(SRC.read_text())
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ("PAGE_NOTICE_SELECTOR", "PAGE_ERROR_SELECTOR", "AUTH_FAILURE_PHRASES"):
            consts[name] = ast.literal_eval(node.value)
assert {"PAGE_NOTICE_SELECTOR", "PAGE_ERROR_SELECTOR", "AUTH_FAILURE_PHRASES"} <= set(consts), consts
AUTH = consts["AUTH_FAILURE_PHRASES"]
NOTICE, ERROR = consts["PAGE_NOTICE_SELECTOR"], consts["PAGE_ERROR_SELECTOR"]
print(f"PAGE_NOTICE_SELECTOR = {NOTICE}")
print(f"PAGE_ERROR_SELECTOR  = {ERROR}\n")

# Reproduces production: banner is fixed-position, covers the whole viewport
# bottom-right where the FAB sits, and never auto-dismisses (lifespan="0").
PAGE = """
<html><body style="margin:0;height:100vh">
  <div>
    <qz-message icon="true" role="alert" lifespan="0" mode="light" notify="true"
                accent="true" status="error" data-drupal-message-type="error"
                data-drupal-message-id="error-892835456951214"
                style="position:fixed;inset:0;background:rgba(255,0,0,.25);display:block;z-index:9999">
      Unrecognized username or password. Have you forgotten your password?
    </qz-message>
  </div>
  <button tabindex="0" type="button" aria-expanded="false"
          data-once="ai-chatbot-toggle" class="fab-mode cursor-pointer"
          aria-label="Open chat assistant"
          style="position:fixed;bottom:20px;right:20px;width:60px;height:60px"
          onclick="window.__fabClicked=true">chat</button>
</body></html>
"""

JS_DISMISS = """(sel) => {
    const els = Array.from(document.querySelectorAll(sel));
    els.forEach(e => e.remove());
    return els.length;
}"""
JS_COLLECT = """(sel) => Array.from(document.querySelectorAll(sel))
     .map(e => (e.innerText || e.textContent || '').trim())
     .filter(t => t.length > 0)"""
JS_CLICK = """() => {
    const b = document.querySelector('button.fab-mode, div.fab-mode');
    if (b) { b.click(); return true; }
    return false;
}"""

failures = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond: failures.append(name)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    print("TEST 1 — reproduce the production failure (no fix)")
    page.set_content(PAGE)
    fab = page.locator("button.fab-mode, div.fab-mode").first
    fab.wait_for(state="visible", timeout=5000)
    try:
        fab.click(timeout=3000)
        check("plain click is intercepted", False, "click unexpectedly succeeded")
    except PWError as e:
        msg = str(e)
        check("plain click is intercepted", "intercepts pointer events" in msg,
              "qz-message intercepts pointer events")

    print("\nTEST 2 — FIX 2: dismiss notices, then click")
    page.set_content(PAGE)
    removed = page.evaluate(JS_DISMISS, NOTICE)
    check("banner removed", removed == 1, f"removed={removed}")
    fab = page.locator("button.fab-mode, div.fab-mode").first
    try:
        fab.click(timeout=3000)
        check("click now succeeds", bool(page.evaluate("() => window.__fabClicked === true")))
    except PWError as e:
        check("click now succeeds", False, str(e)[:80])

    print("\nTEST 3 — FIX 2: JS click works even if the banner is still there")
    page.set_content(PAGE)
    ok = page.evaluate(JS_CLICK)
    check("js_click_fab returns True", ok is True)
    check("handler fired despite overlay",
          bool(page.evaluate("() => window.__fabClicked === true")))

    print("\nTEST 4 — FIX 1: login error is detected and readable")
    page.set_content(PAGE)
    errs = page.evaluate(JS_COLLECT, ERROR)
    check("error notice detected", len(errs) == 1, f"{len(errs)} found")
    check("error text captured",
          bool(errs) and "Unrecognized username or password" in errs[0],
          (errs[0][:60] + "...") if errs else "none")

    print("\nTEST 5 — FIX 1: clean page produces no false positive")
    page.set_content("<html><body><h1>Dashboard</h1></body></html>")
    check("no errors on clean page", page.evaluate(JS_COLLECT, ERROR) == [])

    print("\nTEST 6 — FIX 1: auth-rejection wording IS treated as a login failure")
    for phrase in ["Unrecognized username or password. Have you forgotten your password?",
                   "There have been more than 5 failed login attempts for this account."]:
        check(f"classified as auth failure: {phrase[:38]}...",
              any(p in phrase.lower() for p in AUTH))

    print("\nTEST 7 — FIX 1: unrelated site errors are NOT treated as login failures")
    # Regression guard: on 2026-08-24 a SUCCESSFUL login (HTTP 303) carried an
    # unrelated error banner. Failing on it would abort a working run.
    for phrase in ["An unexpected error has occurred. Please try again later.",
                   "The website encountered an unexpected error.",
                   "Access denied. You do not have permission for this action."]:
        check(f"not an auth failure: {phrase[:38]}...",
              not any(p in phrase.lower() for p in AUTH))

    browser.close()

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}"); sys.exit(1)
print("All checks passed.")
