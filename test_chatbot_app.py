# test_chatbot_app.py
import os
from typing import List, Dict, Any, Optional
import pathlib
import json

import streamlit as st

# =========================
# .env LOADING
# =========================

try:
    from dotenv import load_dotenv

    env_path = pathlib.Path(__file__).parent / ".env"
    load_dotenv(env_path)
except Exception as e:
    print("Failed to load .env:", e)

# =========================
# OPTIONAL CLIENT IMPORTS
# =========================

# Notion client
try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NotionClient = None
    NOTION_AVAILABLE = False

# OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

# =========================
# CONFIG FROM ENV
# =========================

APP_TITLE = "🧪 Testing Chatbot"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")  # MUST be set, no default
NOTION_API_KEY = os.getenv("NOTION_API_KEY") or os.getenv("NOTION_TOKEN")
NOTION_PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID")

# Initialize OpenAI client if possible
openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# NOTION HELPERS
# =========================

def notion_client_from_env() -> NotionClient:
    if not NOTION_AVAILABLE:
        raise RuntimeError("notion-client is not installed. Run: pip install notion-client")
    if not NOTION_API_KEY:
        raise RuntimeError("NOTION_API_KEY / NOTION_TOKEN is not set in environment.")
    return NotionClient(auth=NOTION_API_KEY)


def get_child_pages(parent_page_id: str) -> List[Dict[str, Any]]:
    """Returns child_page blocks under a parent page."""
    client = notion_client_from_env()

    all_blocks: List[Dict[str, Any]] = []
    cursor = None

    while True:
        resp = client.blocks.children.list(
            block_id=parent_page_id,
            start_cursor=cursor,
        )
        all_blocks.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return [b for b in all_blocks if b.get("type") == "child_page"]


def get_latest_child_page_id(parent_page_id: str) -> Optional[str]:
    """Among child pages, find the one with the most recent last_edited_time."""
    client = notion_client_from_env()
    child_pages = get_child_pages(parent_page_id)

    if not child_pages:
        return None

    latest_id = None
    latest_time = None

    for block in child_pages:
        page_id = block["id"]
        page = client.pages.retrieve(page_id=page_id)
        last_edited = page.get("last_edited_time")
        if last_edited is None:
            continue
        if latest_time is None or last_edited > latest_time:
            latest_time = last_edited
            latest_id = page_id

    return latest_id


def load_text_from_page_blocks(page_id: str) -> str:
    """
    Concatenate text from paragraphs & headings of a given page.
    This text will be our Notion-based system prompt content.
    """
    client = notion_client_from_env()

    lines: List[str] = []
    cursor = None

    while True:
        resp = client.blocks.children.list(block_id=page_id, start_cursor=cursor)
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

    return "\n\n".join(lines).strip()


def load_base_prompt_from_notion() -> str:
    """
    Uses NOTION_PARENT_PAGE_ID from env:
    - finds latest child page under it
    - returns its text content as base system prompt.
    """
    if not NOTION_PARENT_PAGE_ID:
        raise RuntimeError("NOTION_PARENT_PAGE_ID is not set in environment.")

    latest_page_id = get_latest_child_page_id(NOTION_PARENT_PAGE_ID)
    if not latest_page_id:
        raise RuntimeError("No child pages found under NOTION_PARENT_PAGE_ID.")

    content = load_text_from_page_blocks(latest_page_id)
    if not content:
        raise RuntimeError("No textual content found in latest child page.")

    return content


# =========================
# JSON PARSING
# =========================

def parse_json_response(reply: str) -> tuple[str, Optional[int]]:
    """
    Parses JSON response from AI in format:
    {
      "message": "user message text",
      "completeness_score": 90
    }
    
    Returns: (message, completeness_score)
    If parsing fails, returns (original_reply, None)
    """
    # Try to parse as JSON
    try:
        # Remove markdown code blocks if present
        clean_reply = reply.strip()
        if clean_reply.startswith("```"):
            # Remove code block markers
            lines = clean_reply.split("\n")
            # Remove first line if it's ```json or ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_reply = "\n".join(lines).strip()
        
        data = json.loads(clean_reply)
        message = data.get("message", "")
        score = data.get("completeness_score")
        
        # Validate score is a number between 1-100
        if score is not None:
            score = int(score)
            if score < 1:
                score = 1
            elif score > 100:
                score = 100
        
        return message, score
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # If parsing fails, return original reply with no score
        return reply, None


# =========================
# CHATBOT BACKEND
# =========================

def build_system_prompt() -> str:
    """
    System prompt = base prompt with placeholders replaced:
      Base prompt is:
        - st.session_state.test_case_prompt (override) if provided, otherwise
        - st.session_state.notion_base_prompt

      Then we replace:
        - {{test_case}}         -> st.session_state.test_case
        - {{persona}}           -> st.session_state.persona
        - {{test_case_details}} -> st.session_state.test_case_details
    """
    override_prompt = st.session_state.get("test_case_prompt", "").strip()
    notion_part = st.session_state.get("notion_base_prompt", "")

    base_prompt = override_prompt or notion_part
    if not base_prompt:
        raise RuntimeError(
            "Base prompt is empty. Either load from Notion or fill 'Test Case Prompt (override)'."
        )

    test_case = st.session_state.get("test_case", "").strip()
    persona = st.session_state.get("persona", "").strip()
    test_case_details = st.session_state.get("test_case_details", "").strip()

    # Keep test_case + persona mandatory (same as before)
    if not test_case or not persona:
        raise RuntimeError("Both 'Test case' and 'Persona' must be filled in the sidebar.")

    # Replace placeholders (details can be empty, that's fine)
    system_prompt = (
        base_prompt
        .replace("{{test_case}}", test_case)
        .replace("{{persona}}", persona)
        .replace("{{test_case_details}}", test_case_details)
    )

    return system_prompt


def call_chatbot(user_message: str, history: List[Dict[str, str]]) -> tuple[str, Optional[int]]:
    """
    Uses OpenAI with:
    - system: base prompt (Notion or override) with placeholders resolved
    - user/assistant: chat history + latest input

    Returns: (message, completeness_score)
    
    Also stores debug info in st.session_state:
      - debug_openai_messages
      - debug_openai_raw_response
    """
    if openai_client is None:
        return "❌ OPENAI_API_KEY not configured or OpenAI client unavailable.", None

    if not OPENAI_MODEL:
        return "❌ OPENAI_MODEL not set in environment.", None

    try:
        system_prompt = build_system_prompt()
    except Exception as e:
        return f"❌ System prompt error: {e}", None

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": user_message})

    # Store request messages in debug state
    st.session_state.debug_openai_messages = messages

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )

        # Extract assistant reply
        reply = resp.choices[0].message.content

        # Parse JSON response
        message, score = parse_json_response(reply)

        # Store raw response in debug state (as dict if possible)
        try:
            raw_dict = resp.model_dump(exclude_none=True)
        except Exception:
            # Fallback: store string representation
            raw_dict = {"raw": str(resp)}

        st.session_state.debug_openai_raw_response = raw_dict

        return message, score

    except Exception as e:
        st.session_state.debug_openai_raw_response = {"error": str(e)}
        return f"❌ OpenAI error: {e}", None


# =========================
# STREAMLIT APP
# =========================

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)


# Init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []

if "score_history" not in st.session_state:
    st.session_state.score_history: List[int] = []

if "notion_base_prompt" not in st.session_state:
    st.session_state.notion_base_prompt = ""
if "notion_error" not in st.session_state:
    st.session_state.notion_error = None
if "test_case" not in st.session_state:
    st.session_state.test_case = ""
if "persona" not in st.session_state:
    st.session_state.persona = ""
if "test_case_details" not in st.session_state:
    st.session_state.test_case_details = ""
if "test_case_prompt" not in st.session_state:
    st.session_state.test_case_prompt = ""
if "debug_openai_messages" not in st.session_state:
    st.session_state.debug_openai_messages = None
if "debug_openai_raw_response" not in st.session_state:
    st.session_state.debug_openai_raw_response = None

# Sidebar
with st.sidebar:
    st.subheader("Environment Status")
    st.write("**OPENAI_API_KEY:** " + (f"{OPENAI_API_KEY[:4]}..." if OPENAI_API_KEY else "❌ missing"))
    st.write("**OPENAI_MODEL:** " + (f"{OPENAI_MODEL}" if OPENAI_MODEL else "❌ missing"))
    st.write("**NOTION_API_KEY:** " + (f"{NOTION_API_KEY[:4]}..." if NOTION_API_KEY else "❌ missing"))
    st.write("**NOTION_PARENT_PAGE_ID:** " + (f"{NOTION_PARENT_PAGE_ID[:8]}..." if NOTION_PARENT_PAGE_ID else "❌ missing"))

    st.markdown("---")

    st.subheader("Placeholders")
    st.session_state.test_case = st.text_area(
        "Test case ({{test_case}})",
        value=st.session_state.test_case,
        placeholder="Onboarding & \"How it works\"",
        height=80,
    )
    st.session_state.persona = st.text_area(
        "Persona ({{persona}})",
        value=st.session_state.persona,
        placeholder="HOST or auPair",
        height=110,
    )
    st.session_state.test_case_details = st.text_area(
        "Test case details ({{test_case_details}})",
        value=st.session_state.test_case_details,
        placeholder="Additional scenario details, edge cases, constraints...",
        height=120,
    )
    st.session_state.test_case_prompt = st.text_area(
        "Test Case Prompt (override)",
        value=st.session_state.test_case_prompt,
        placeholder="Optional: if filled, this replaces the Notion base prompt for this run.",
        help="This field is automatically filled when running via Playwright bridge with prompt from Test Case Executions DB",
        height=140,
    )

    st.markdown("---")

    if st.button("🔄 Reload base prompt from Notion"):
        try:
            base_prompt = load_base_prompt_from_notion()
            st.session_state.notion_base_prompt = base_prompt
            st.session_state.notion_error = None
        except Exception as e:
            st.session_state.notion_base_prompt = ""
            st.session_state.notion_error = str(e)

    if st.button("🧹 Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.score_history = []
        st.session_state.debug_openai_messages = None
        st.session_state.debug_openai_raw_response = None
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Completeness Score")
    
    if st.session_state.score_history:
        current_score = st.session_state.score_history[-1]
        
        # Color-code the score
        if current_score >= 80:
            score_color = "🟢"
            score_status = "Excellent"
        elif current_score >= 60:
            score_color = "🟡"
            score_status = "Good"
        elif current_score >= 40:
            score_color = "🟠"
            score_status = "Fair"
        else:
            score_color = "🔴"
            score_status = "Needs Work"
        
        st.markdown(f"### {score_color} {current_score}/100")
        st.caption(score_status)
        
        # Show progress
        if len(st.session_state.score_history) > 1:
            st.markdown("**Score Progression:**")
            
            # Create a simple line chart
            import pandas as pd
            df = pd.DataFrame({
                'Turn': list(range(1, len(st.session_state.score_history) + 1)),
                'Score': st.session_state.score_history
            })
            st.line_chart(df.set_index('Turn'))
            
            # Show change from previous
            prev_score = st.session_state.score_history[-2]
            change = current_score - prev_score
            if change > 0:
                st.caption(f"↗️ +{change} from previous turn")
            elif change < 0:
                st.caption(f"↘️ {change} from previous turn")
            else:
                st.caption("➡️ No change from previous turn")
    else:
        st.info("No scores yet. Start chatting to see completeness tracking.")

    st.markdown("---")
    st.subheader("Raw Notion base prompt")
    if st.session_state.notion_error:
        st.error(f"Error loading Notion prompt: {st.session_state.notion_error}")
    elif st.session_state.notion_base_prompt:
        with st.expander("View Notion prompt", expanded=False):
            st.code(st.session_state.notion_base_prompt, language="text")
    else:
        st.info("Notion base prompt not loaded yet. Click **Reload base prompt from Notion**.")

    st.markdown("---")
    st.subheader("Effective system prompt")

    base_prompt_available = (
        bool(st.session_state.notion_base_prompt) or
        bool(st.session_state.test_case_prompt)
    )
    if base_prompt_available:
        try:
            resolved_prompt = build_system_prompt()
            with st.expander("View system prompt", expanded=False):
                st.code(resolved_prompt, language="text")
        except Exception as e:
            st.info(
                "System prompt is not ready yet. "
                "Make sure **Test case** and **Persona** are filled above.\n\n"
                f"Details: {e}"
            )
    else:
        st.info(
            "No base prompt available. Either:\n"
            "• Load from Notion (Raw Notion base prompt), or\n"
            "• Fill 'Test Case Prompt (override)'."
        )

    st.markdown("---")
    st.subheader("Debug: Last OpenAI call")

    debug_msgs = st.session_state.get("debug_openai_messages", None)
    debug_raw  = st.session_state.get("debug_openai_raw_response", None)

    if debug_msgs is not None:
        with st.expander("Request messages", expanded=False):
            st.json(debug_msgs)
    else:
        st.info("No OpenAI request yet. Send a message in the chat.")

    if debug_raw is not None:
        with st.expander("Raw response", expanded=False):
            st.json(debug_raw)
    else:
        st.info("No OpenAI response yet. Send a message in the chat.")


# ✅ Auto-load base prompt on first run (optional)
# ONLY do this if we don't already have a Test Case Prompt override
# (which would be filled by the Playwright bridge from Test Case Executions DB)
if (not st.session_state.notion_base_prompt and 
    not st.session_state.notion_error and
    not st.session_state.test_case_prompt):  # ← Check if override is empty
    try:
        base_prompt = load_base_prompt_from_notion()
        st.session_state.notion_base_prompt = base_prompt
    except Exception as e:
        st.session_state.notion_error = str(e)

# Show conversation history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["assistant"])

# Chat input (normal user messages)
user_input = st.chat_input("Type your message...")

if user_input is not None and user_input.strip():
    user_text = user_input.strip()

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_text)

    # Call chatbot with history + system prompt (Notion or override)
    reply, score = call_chatbot(user_text, st.session_state.chat_history)

    # Show reply
    with st.chat_message("assistant"):
        # Display the message
        st.markdown(reply)
        
        # Display the score if available
        if score is not None:
            # Color-code based on score
            if score >= 80:
                score_badge = f"🟢 **Completeness: {score}/100** (Excellent)"
            elif score >= 60:
                score_badge = f"🟡 **Completeness: {score}/100** (Good)"
            elif score >= 40:
                score_badge = f"🟠 **Completeness: {score}/100** (Fair)"
            else:
                score_badge = f"🔴 **Completeness: {score}/100** (Needs Work)"
            
            st.markdown(f"---\n{score_badge}")

    # Save to history
    st.session_state.chat_history.append(
        {"user": user_text, "assistant": reply}
    )
    
    # Save score if available
    if score is not None:
        st.session_state.score_history.append(score)

    # 🔁 Force a rerun so the sidebar shows updated score
    st.rerun()