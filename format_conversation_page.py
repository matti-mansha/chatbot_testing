# format_conversation_page.py
import os
import re
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
import httpx

from logging_config import setup_logging, log_exception, log_api_call

# =====================================
# LOAD .env
# =====================================

BASE_DIR = pathlib.Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# Set up logging
logger = setup_logging("format_conversation_page")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
CONVERSATIONS_PARENT_PAGE_ID = os.getenv("CONVERSATIONS_PARENT_PAGE_ID")

NOTION_API_BASE = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


# =====================================
# CONVERSATION PARSER
# =====================================

# System / status emoji prefixes to skip during message collection
_SYSTEM_EMOJI_PREFIXES = (
    '⏳', '✓', '➡️', '⚠️', '❌', '📝', '🔐', '🔓', '💬', '🗑️',
    '✅', '✨', '📊', '🔄', '🚨', '🧹', 'ℹ️', '🎯',
)

# Lines matching these patterns mark the end of conversation content
_POST_CONVERSATION_PATTERNS = [
    'Chat loop finished',
    'TEST COMPLETED',
    'Diagnostics saved',
    'Summary:',
    'MAX RESTARTS EXCEEDED',
    'Test failed to complete',
    'Unexpected error',
    'RESTARTING ENTIRE TEST',
    'Conversation entries:',
    'Number of turns:',
]


def _extract_score(message_text: str):
    """Extract completeness score from message text and return (cleaned_text, score)."""
    score_match = re.search(r'[🟢🟡🟠🔴]\s*\*\*Completeness:\s*(\d+)/100\*\*', message_text)
    if score_match:
        score = int(score_match.group(1))
        cleaned = re.sub(
            r'\n*---\s*\n*[🟢🟡🟠🔴]\s*\*\*Completeness:\s*\d+/100\*\*.*$',
            '', message_text, flags=re.MULTILINE
        ).strip()
        return cleaned, score
    return message_text, None


def _is_noise_line(line: str) -> bool:
    """Return True if the line is system noise that should be skipped."""
    stripped = line.strip()
    if not stripped:
        return False  # blank lines are OK inside messages
    # Pure separator lines (5+ equals signs with nothing else)
    if re.match(r'^[=\-]{5,}\s*$', stripped):
        return True
    # System emoji prefixes
    if stripped.startswith(_SYSTEM_EMOJI_PREFIXES):
        return True
    # Post-conversation markers
    if any(pat in stripped for pat in _POST_CONVERSATION_PATTERNS):
        return True
    # Turn count lines like "Turns: 5", "Restart attempts: 0"
    if re.match(r'^\s*(Turns|Restart attempts|Completed \d+ turns)', stripped):
        return True
    return False


def _save_message(current_speaker, current_message_lines, current_turn, turns):
    """Flush the current message into the current turn. Returns updated (current_turn, turns)."""
    if not current_speaker or not current_message_lines:
        return current_turn, turns

    message_text = '\n'.join(current_message_lines).strip()
    if not message_text or message_text == 'None':
        return current_turn, turns

    message_text, score = _extract_score(message_text)

    if current_turn is None:
        current_turn = {'turn_number': 0, 'messages': []}

    current_turn['messages'].append({
        'speaker': current_speaker,
        'text': message_text,
        'score': score,
    })
    return current_turn, turns


def _is_turn_marker(line: str):
    """
    Detect turn markers in two formats:
      1) Single-line:  "========== TURN 3/10 =========="
      2) Standalone:   "TURN 3/10"  (with ===== on surrounding lines)
    Returns the turn number (int) or None.
    """
    stripped = line.strip()
    # Format 1: equals signs + TURN on same line
    m = re.match(r'=+\s*TURN\s+(\d+)(?:/\d+)?\s*=*$', stripped)
    if m:
        return int(m.group(1))
    # Format 2: standalone "TURN N" or "TURN N/M"
    m = re.match(r'^TURN\s+(\d+)(?:/\d+)?$', stripped)
    if m:
        return int(m.group(1))
    return None


BRIDGE_ATTEMPT_BOUNDARY = "===BRIDGE_ATTEMPT_BOUNDARY==="


def _trim_to_last_attempt(text: str) -> str:
    """
    When the bridge bot restarts mid-run (e.g. MilaBackendRejection), its
    subprocess stdout accumulates output from ALL attempts — the final
    conversation page would otherwise show duplicate/out-of-order turns like
    "1, 2, 3, 4, 5, 1".

    The bridge prints `===BRIDGE_ATTEMPT_BOUNDARY===` at the start of each
    attempt. We keep only the content after the LAST boundary (= the final
    attempt). Any bridge-failure banner prepended by run_test_executions.py
    appears BEFORE the first boundary and is preserved by carrying it across
    the trim. If no boundary exists (older bridge versions, or success-only
    successful runs before the boundary feature), we return the text
    unchanged.
    """
    last = text.rfind(BRIDGE_ATTEMPT_BOUNDARY)
    if last == -1:
        return text

    pre = text[:last]
    post = text[last + len(BRIDGE_ATTEMPT_BOUNDARY):]

    # Preserve the "⚠️ BRIDGE FAILURE / ... / --- Partial transcript below ---"
    # banner if run_test_executions.py prepended one. It's diagnostic content
    # that the user wants to see on failed pages.
    import re as _re
    banner_match = _re.search(
        r'(⚠️ BRIDGE FAILURE.*?--- Partial transcript below[^\n]*---\n\n)',
        pre,
        _re.DOTALL,
    )
    banner = banner_match.group(1) if banner_match else ''

    trimmed = banner + post
    logger.debug(
        f"Trimmed stdout to final attempt: {len(text)} → {len(trimmed)} chars "
        f"(banner preserved: {bool(banner)})"
    )
    return trimmed


def parse_conversation_transcript(text: str) -> Dict[str, Any]:
    """
    Parse the conversation transcript into structured sections.
    Returns a dict with metadata and turns.
    """
    logger.debug(f"Parsing conversation transcript ({len(text)} chars)")
    text = _trim_to_last_attempt(text)
    lines = text.split('\n')

    # ---- find conversation boundaries ----
    # Start: first speaker marker or first TURN marker
    conv_start = 0
    for i, line in enumerate(lines):
        if line.startswith('🟠 Mila') or _is_turn_marker(line) is not None:
            conv_start = i
            break

    # End: first post-conversation marker after conversation starts
    conv_end = len(lines)
    for i in range(conv_start, len(lines)):
        stripped = lines[i].strip()
        if 'Chat loop finished' in stripped or 'TEST COMPLETED' in stripped:
            conv_end = i
            break

    metadata_text = '\n'.join(lines[:conv_start]).strip()
    conv_lines = lines[conv_start:conv_end]

    logger.debug(f"Conversation boundaries: lines {conv_start}-{conv_end} "
                 f"({len(conv_lines)} lines), metadata {conv_start} lines")

    # ---- parse turns ----
    turns: List[Dict[str, Any]] = []
    current_turn = None
    current_speaker = None
    current_message_lines: List[str] = []

    for line in conv_lines:
        # Turn marker
        turn_number = _is_turn_marker(line)
        if turn_number is not None:
            current_turn, turns = _save_message(
                current_speaker, current_message_lines, current_turn, turns)
            current_message_lines = []
            current_speaker = None

            # Save previous turn
            if current_turn:
                turns.append(current_turn)

            current_turn = {'turn_number': turn_number, 'messages': []}
            logger.debug(f"Starting turn {turn_number}")
            continue

        # Speaker markers
        if line.startswith('🟠 Mila'):
            current_turn, turns = _save_message(
                current_speaker, current_message_lines, current_turn, turns)
            current_message_lines = []
            current_speaker = 'Mila'
            continue

        if line.startswith('🧪 Tester'):
            current_turn, turns = _save_message(
                current_speaker, current_message_lines, current_turn, turns)
            current_message_lines = []
            current_speaker = 'Tester'
            continue

        # Skip system / noise lines
        if _is_noise_line(line):
            continue

        # Collect message content
        if current_speaker:
            current_message_lines.append(line)

    # Flush final message & turn
    current_turn, turns = _save_message(
        current_speaker, current_message_lines, current_turn, turns)
    if current_turn:
        turns.append(current_turn)

    # ---- fold Turn 0 into Turn 1 ----
    # The opening Mila message lands in a synthetic Turn 0 because it appears
    # before the first TURN marker. Merge it into Turn 1 for a cleaner layout.
    if len(turns) >= 2 and turns[0]['turn_number'] == 0 and turns[1]['turn_number'] == 1:
        turns[1]['messages'] = turns[0]['messages'] + turns[1]['messages']
        turns.pop(0)
    elif len(turns) == 1 and turns[0]['turn_number'] == 0:
        turns[0]['turn_number'] = 1

    logger.info(f"Parsed {len(turns)} turns from conversation")

    return {
        'metadata': metadata_text,
        'turns': turns,
    }


# =====================================
# NOTION BLOCK BUILDERS
# =====================================

def create_heading_block(text: str, level: int = 2) -> Dict[str, Any]:
    """Create a Notion heading block."""
    heading_type = f"heading_{level}"
    return {
        "object": "block",
        "type": heading_type,
        heading_type: {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def create_divider_block() -> Dict[str, Any]:
    """Create a Notion divider block."""
    return {
        "object": "block",
        "type": "divider",
        "divider": {}
    }


def create_callout_block(text: str, icon: str = "💬", color: str = "gray_background") -> Dict[str, Any]:
    """Create a Notion callout block with icon and color."""
    # Split long text into chunks
    chunks = split_text_for_notion(text, max_length=1900)
    
    # Return first chunk as callout
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": icon},
            "color": color,
            "rich_text": [
                {"type": "text", "text": {"content": chunks[0]}}
            ]
        }
    }


def create_paragraph_block(text: str) -> Dict[str, Any]:
    """Create a Notion paragraph block."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def create_toggle_block(title: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a Notion toggle (collapsible) block."""
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": title}}],
            "children": children
        }
    }


def create_toggle_blocks_with_limit(title: str, children: List[Dict[str, Any]], max_children: int = 100) -> List[Dict[str, Any]]:
    """
    Create one or more toggle blocks, splitting children if they exceed Notion's 100-child limit.
    Returns a list of toggle blocks.
    """
    if len(children) <= max_children:
        return [create_toggle_block(title, children)]
    
    toggle_blocks = []
    total_parts = (len(children) + max_children - 1) // max_children
    
    for i in range(0, len(children), max_children):
        chunk = children[i:i + max_children]
        part_num = (i // max_children) + 1
        chunk_title = f"{title} (Part {part_num}/{total_parts})" if total_parts > 1 else title
        toggle_blocks.append(create_toggle_block(chunk_title, chunk))
    
    return toggle_blocks

def split_text_for_notion(text: str, max_length: int = 2000) -> List[str]:
    """Split a long text into chunks for Notion (max chars per chunk)."""
    if not text:
        return [""]
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


# =====================================
# IMPROVED CONVERSATION PAGE CREATOR
# =====================================

def _score_badge(score: int) -> tuple:
    """Return (icon, label, notion_color) for a completeness score."""
    if score >= 80:
        return "🟢", "Excellent", "green_background"
    if score >= 60:
        return "🟡", "Good", "yellow_background"
    if score >= 40:
        return "🟠", "Fair", "orange_background"
    return "🔴", "Needs Work", "red_background"


def _build_turn_children(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the Notion child blocks for a single turn (inside a toggle)."""
    children: List[Dict[str, Any]] = []

    for msg in turn['messages']:
        speaker = msg['speaker']
        text = msg.get('text', '')
        if not text or text == 'None':
            continue

        icon = "🤖" if speaker == 'Mila' else "🧪"
        color = "blue_background" if speaker == 'Mila' else "green_background"

        chunks = split_text_for_notion(text, max_length=1900)
        children.append(create_callout_block(chunks[0], icon=icon, color=color))
        for chunk in chunks[1:]:
            children.append(create_paragraph_block(chunk))

    return children


def create_formatted_conversation_page(
    parent_page_id: str,
    title: str,
    conversation_text: str,
) -> Optional[str]:
    """
    Create a formatted Notion page with the conversation transcript.
    Returns the created page ID or None on error.
    """
    logger.info(f"Creating formatted conversation page: {title}")
    logger.debug(f"Parent page: {parent_page_id}")
    logger.debug(f"Conversation length: {len(conversation_text)} chars")

    if not conversation_text:
        conversation_text = "(No conversation captured)"
        logger.warning("Empty conversation text provided")

    parent_uuid = format_notion_id(parent_page_id)
    parsed = parse_conversation_transcript(conversation_text)

    blocks: List[Dict[str, Any]] = []

    # ---- Completeness Score Summary (top of page) ----
    scores = []
    for turn in parsed['turns']:
        for msg in turn['messages']:
            if msg.get('score') is not None:
                scores.append((turn['turn_number'], msg['score']))

    if scores:
        score_values = [s for _, s in scores]
        final_score = score_values[-1]
        avg_score = sum(score_values) / len(score_values)

        logger.info(f"Score statistics: Final={final_score}, Avg={avg_score:.1f}, "
                     f"Min={min(score_values)}, Max={max(score_values)}")

        s_icon, s_label, s_color = _score_badge(final_score)

        blocks.append(create_heading_block("Completeness Score", level=2))
        blocks.append(create_callout_block(
            f"{s_icon}  Final Score: {final_score}/100  —  {s_label}",
            icon="🎯", color=s_color,
        ))

        stats_parts = [
            f"Avg {avg_score:.0f}",
            f"Min {min(score_values)}",
            f"Max {max(score_values)}",
            f"{len(score_values)} turns scored",
        ]
        blocks.append(create_paragraph_block("  •  ".join(stats_parts)))

        if len(score_values) > 1:
            blocks.append(create_paragraph_block(
                " → ".join(str(s) for s in score_values)
            ))

        blocks.append(create_divider_block())

    # ---- Conversation Turns (each as a toggle) ----
    logger.debug(f"Processing {len(parsed['turns'])} turns")
    for turn in parsed['turns']:
        turn_num = turn['turn_number']

        # Build toggle title with score badge if available
        turn_score = None
        for msg in turn['messages']:
            if msg.get('score') is not None:
                turn_score = msg['score']
        if turn_score is not None:
            icon, label, _ = _score_badge(turn_score)
            toggle_title = f"Turn {turn_num}  —  {icon} {turn_score}/100"
        else:
            toggle_title = f"Turn {turn_num}"

        children = _build_turn_children(turn)
        if not children:
            continue

        # Notion limits toggle children to 100
        toggle_blocks = create_toggle_blocks_with_limit(toggle_title, children)
        blocks.extend(toggle_blocks)

    # ---- Fallback if nothing parsed ----
    if not blocks:
        logger.warning("Could not parse conversation structure, using simple format")
        paragraphs = conversation_text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            for chunk in split_text_for_notion(para, max_length=1800):
                blocks.append(create_paragraph_block(chunk))

    if not blocks:
        blocks.append(create_paragraph_block("(Empty conversation)"))
        logger.warning("No blocks created, adding empty conversation message")

    logger.debug(f"Created {len(blocks)} blocks for conversation page")
    
    # Create the page
    payload = {
        "parent": {"page_id": parent_uuid},
        "properties": {
            "title": [{"type": "text", "text": {"content": title}}]
        },
        "children": blocks
    }

    try:
        start_time = datetime.now()
        resp = httpx.post(
            f"{NOTION_API_BASE}/pages",
            headers=HEADERS,
            json=payload,
            timeout=60.0,
        )
        duration = (datetime.now() - start_time).total_seconds()
        log_api_call(logger, "POST", f"{NOTION_API_BASE}/pages", resp.status_code, duration)
        
        resp.raise_for_status()
        page_data = resp.json()
        conv_page_id = page_data.get("id")
        
        logger.info(f"✓ Created formatted conversation page: {conv_page_id}")
        print(f"   ✓ Created formatted conversation page: {conv_page_id}")
        return conv_page_id
        
    except Exception as e:
        log_exception(logger, e, "create_formatted_conversation_page")
        logger.error(f"❌ Error creating conversation page: {e}")
        print(f"❌ Error creating conversation page: {e}")
        try:
            logger.debug(f"Response: {resp.text}")
            print("   Response:", resp.text)
        except Exception:
            pass
        return None


def format_notion_id(notion_id: str) -> str:
    """Convert a Notion ID to UUID format with hyphens if it is 32 chars."""
    clean_id = (notion_id or "").replace("-", "")
    if len(clean_id) == 32:
        return f"{clean_id[0:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:32]}"
    return notion_id


# =====================================
# STANDALONE USAGE
# =====================================

def main():
    """
    Standalone usage: Read a conversation file and create a formatted Notion page.
    Usage: python format_conversation_page.py <conversation_file.txt> [page_title]
    """
    import sys
    
    logger.info("=" * 80)
    logger.info("STANDALONE MODE - format_conversation_page.py")
    logger.info("=" * 80)
    
    if len(sys.argv) < 2:
        print("Usage: python format_conversation_page.py <conversation_file.txt> [page_title]")
        print("\nOr import this module and use create_formatted_conversation_page() function")
        logger.info("No arguments provided, showing usage")
        return
    
    conversation_file = sys.argv[1]
    page_title = sys.argv[2] if len(sys.argv) > 2 else f"Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    logger.info(f"Conversation file: {conversation_file}")
    logger.info(f"Page title: {page_title}")
    
    if not os.path.exists(conversation_file):
        logger.error(f"❌ File not found: {conversation_file}")
        print(f"❌ File not found: {conversation_file}")
        return
    
    if not NOTION_API_KEY or not CONVERSATIONS_PARENT_PAGE_ID:
        logger.error("❌ Missing NOTION_API_KEY or CONVERSATIONS_PARENT_PAGE_ID in .env")
        print("❌ Missing NOTION_API_KEY or CONVERSATIONS_PARENT_PAGE_ID in .env")
        return
    
    with open(conversation_file, 'r', encoding='utf-8') as f:
        conversation_text = f.read()
    
    logger.info(f"Read {len(conversation_text)} characters from file")
    print(f"📄 Creating formatted conversation page: {page_title}")
    
    page_id = create_formatted_conversation_page(
        parent_page_id=CONVERSATIONS_PARENT_PAGE_ID,
        title=page_title,
        conversation_text=conversation_text
    )
    
    if page_id:
        page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
        logger.info(f"✅ Success! Page URL: {page_url}")
        print(f"✅ Success! View your page at: {page_url}")
    else:
        logger.error("❌ Failed to create page")
        print("❌ Failed to create page")


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