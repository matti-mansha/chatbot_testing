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

def parse_conversation_transcript(text: str) -> Dict[str, Any]:
    """
    Parse the conversation transcript into structured sections.
    Returns a dict with metadata and turns.
    """
    logger.debug(f"Parsing conversation transcript ({len(text)} chars)")
    lines = text.split('\n')
    
    # Extract metadata (everything before first turn)
    metadata_lines = []
    conversation_start_idx = 0
    
    for i, line in enumerate(lines):
        if '========== TURN' in line:
            conversation_start_idx = i
            break
        metadata_lines.append(line)
    
    logger.debug(f"Found {len(metadata_lines)} metadata lines, conversation starts at line {conversation_start_idx}")
    
    # Parse turns
    turns = []
    current_turn = None
    current_speaker = None
    current_message_lines = []
    
    for line in lines[conversation_start_idx:]:
        # Turn marker
        turn_match = re.match(r'=+\s*TURN\s+(\d+)\s*=+', line)
        if turn_match:
            # Save previous message if exists
            if current_speaker and current_message_lines:
                # Initialize turn if needed (for conversations without turn markers)
                if current_turn is None:
                    current_turn = {'turn_number': 0, 'messages': []}
                
                # Parse the message and extract score if present
                message_text = '\n'.join(current_message_lines).strip()
                score = None
                
                # Extract completeness score if present (format: "🟢 **Completeness: 90/100**")
                score_match = re.search(r'[🟢🟡🟠🔴]\s*\*\*Completeness:\s*(\d+)/100\*\*', message_text)
                if score_match:
                    score = int(score_match.group(1))
                    # Remove the score line from message text for cleaner display
                    message_text = re.sub(r'---\s*[🟢🟡🟠🔴]\s*\*\*Completeness:\s*\d+/100\*\*.*$', '', message_text, flags=re.MULTILINE).strip()
                
                current_turn['messages'].append({
                    'speaker': current_speaker,
                    'text': message_text,
                    'score': score
                })
                current_message_lines = []
                current_speaker = None
            
            # Save previous turn if exists
            if current_turn:
                turns.append(current_turn)
            
            # Start new turn
            turn_number = int(turn_match.group(1))
            current_turn = {
                'turn_number': turn_number,
                'messages': []
            }
            logger.debug(f"Starting turn {turn_number}")
            continue
        
        # Speaker markers
        if line.startswith('🟠 Mila'):
            # Save previous message if exists
            if current_speaker and current_message_lines:
                message_text = '\n'.join(current_message_lines).strip()
                score = None
                
                # Extract completeness score if present
                score_match = re.search(r'[🟢🟡🟠🔴]\s*\*\*Completeness:\s*(\d+)/100\*\*', message_text)
                if score_match:
                    score = int(score_match.group(1))
                    message_text = re.sub(r'---\s*[🟢🟡🟠🔴]\s*\*\*Completeness:\s*\d+/100\*\*.*$', '', message_text, flags=re.MULTILINE).strip()
                
                # Initialize turn if needed (for conversations without turn markers)
                if current_turn is None:
                    current_turn = {'turn_number': 0, 'messages': []}
                
                current_turn['messages'].append({
                    'speaker': current_speaker,
                    'text': message_text,
                    'score': score
                })
                current_message_lines = []
            
            current_speaker = 'Mila'
            continue
        
        if line.startswith('🧪 Tester'):
            # Save previous message if exists
            if current_speaker and current_message_lines:
                message_text = '\n'.join(current_message_lines).strip()
                score = None
                
                # Extract completeness score if present
                score_match = re.search(r'[🟢🟡🟠🔴]\s*\*\*Completeness:\s*(\d+)/100\*\*', message_text)
                if score_match:
                    score = int(score_match.group(1))
                    message_text = re.sub(r'---\s*[🟢🟡🟠🔴]\s*\*\*Completeness:\s*\d+/100\*\*.*$', '', message_text, flags=re.MULTILINE).strip()
                
                # Initialize turn if needed (for conversations without turn markers)
                if current_turn is None:
                    current_turn = {'turn_number': 0, 'messages': []}
                
                current_turn['messages'].append({
                    'speaker': current_speaker,
                    'text': message_text,
                    'score': score
                })
                current_message_lines = []
            
            current_speaker = 'Tester'
            continue
        
        # System messages (skip these for main content)
        if line.startswith(('⏳', '✓', '➡️', '⚠️', '❌', '📝', '🔐', '🔓', '💬', '🗑️')):
            continue
        
        # Regular message content
        if current_speaker:
            current_message_lines.append(line)
    
    # Save final message and turn
    if current_speaker and current_message_lines:
        # Initialize turn if needed (for conversations without turn markers)
        if current_turn is None:
            current_turn = {'turn_number': 0, 'messages': []}
        
        message_text = '\n'.join(current_message_lines).strip()
        score = None
        
        # Extract completeness score if present
        score_match = re.search(r'[🟢🟡🟠🔴]\s*\*\*Completeness:\s*(\d+)/100\*\*', message_text)
        if score_match:
            score = int(score_match.group(1))
            message_text = re.sub(r'---\s*[🟢🟡🟠🔴]\s*\*\*Completeness:\s*\d+/100\*\*.*$', '', message_text, flags=re.MULTILINE).strip()
        
        current_turn['messages'].append({
            'speaker': current_speaker,
            'text': message_text,
            'score': score
        })
    if current_turn:
        turns.append(current_turn)
    
    logger.info(f"Parsed {len(turns)} turns from conversation")
    
    return {
        'metadata': '\n'.join(metadata_lines).strip(),
        'turns': turns
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

def create_formatted_conversation_page(
    parent_page_id: str,
    title: str,
    conversation_text: str,
) -> Optional[str]:
    """
    Create a beautifully formatted Notion page with the conversation transcript.
    Returns the created page ID or None on error.
    """
    logger.info(f"Creating formatted conversation page: {title}")
    logger.debug(f"Parent page: {parent_page_id}")
    logger.debug(f"Conversation length: {len(conversation_text)} chars")
    
    if not conversation_text:
        conversation_text = "(No conversation captured)"
        logger.warning("Empty conversation text provided")

    parent_uuid = format_notion_id(parent_page_id)
    
    # Parse the conversation
    parsed = parse_conversation_transcript(conversation_text)
    
    # Build blocks
    blocks: List[Dict[str, Any]] = []
    
    # =====================================
    # CLEAN METADATA SECTION - MODIFIED
    # =====================================
    if parsed['metadata']:
        metadata_blocks = []
        
        # Define patterns to SKIP (technical noise)
        skip_patterns = [
            'DOM snapshot', 'Console Logs', 'Network Events',
            'JavaScript Errors', 'UI State Snapshots', 'Message bubble',
            'Error indicators', 'DIAGNOSTIC REPORT', '===',
            'Session ID:', 'Generated:', 'Requests:', 'Responses:',
            'Status Codes:', 'waiting for', 'Waiting for',
            'Found menu', 'Found Clear', 'Clicked', 'clicking',
            'filling', 'Filling', 'login', 'Login', 'cookie',
            'Cookie', 'Saved', 'saved', 'diagnostics/',
            'selector:', 'bubbles:', 'typing', 'complete_dom',
            'sent_dom', 'health check', 'session:', 'API', 'api',
            '.html', '.json', 'REPORT.txt', 'Opening', 'opening',
            'Clearing', 'clearing', 'Creating', 'creating',
            'Activated', 'activated', 'listeners', 'snapshot',
            'Active sessions', 'OpenAI', 'model:', 'time:',
            'Preview:', 'preview:', 'Length:', 'length:',
            'attempt complete', 'send', 'Send', 'input',
            'Input', 'bubble appeared', 'finished typing',
            'new message', 'reply:', 'Tester API',
            'Diagnostic', 'bytes', 'Bytes', 'Check #'
        ]
        
        # Filter metadata to essential lines only
        for line in parsed['metadata'].split('\n'):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Skip lines with technical patterns
            if any(skip in line for skip in skip_patterns):
                continue
            
            # Keep lines that look like important summary info
            if any(keyword in line for keyword in [
                'RESTART', 'TEST COMPLETED', 'Turns:', 'Restart attempts',
                'Early exit', 'EARLY EXIT', 'score='
            ]):
                metadata_blocks.append(create_paragraph_block(line))
        
        # Add filtered metadata if we have any essential info
        if metadata_blocks:
            blocks.append(create_toggle_block("📋 Test Summary", metadata_blocks))
            blocks.append(create_divider_block())
            logger.debug(f"Added filtered metadata with {len(metadata_blocks)} essential lines")
        else:
            logger.debug("No essential metadata to display (all filtered out)")
    
    # =====================================
    # END OF CLEAN METADATA SECTION
    # =====================================
    
    # Add completeness score summary if scores are present
    scores = []
    for turn in parsed['turns']:
        for msg in turn['messages']:
            if msg.get('score') is not None:
                scores.append(msg['score'])
    
    if scores:
        # Calculate statistics
        final_score = scores[-1]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        logger.info(f"Score statistics: Final={final_score}, Avg={avg_score:.1f}, Min={min_score}, Max={max_score}")
        
        # Determine overall status
        if final_score >= 80:
            status_icon = "🟢"
            status_text = "Excellent"
            status_color = "green_background"
        elif final_score >= 60:
            status_icon = "🟡"
            status_text = "Good"
            status_color = "yellow_background"
        elif final_score >= 40:
            status_icon = "🟠"
            status_text = "Fair"
            status_color = "orange_background"
        else:
            status_icon = "🔴"
            status_text = "Needs Work"
            status_color = "red_background"
        
        # Add summary heading
        blocks.append(create_heading_block("📊 Completeness Score Summary", level=2))
        
        # Add final score callout
        final_score_text = f"{status_icon} Final Score: {final_score}/100 ({status_text})"
        blocks.append(create_callout_block(final_score_text, icon="🎯", color=status_color))
        
        # Add statistics
        stats_text = f"Average: {avg_score:.1f}/100  •  Min: {min_score}/100  •  Max: {max_score}/100  •  Total Turns: {len(scores)}"
        blocks.append(create_paragraph_block(stats_text))
        
        # Add score progression if more than one score
        if len(scores) > 1:
            progression_text = "Score Progression: " + " → ".join([str(s) for s in scores])
            blocks.append(create_paragraph_block(progression_text))
        
        blocks.append(create_divider_block())
    
    # Add conversation turns
    logger.debug(f"Processing {len(parsed['turns'])} turns")
    for turn in parsed['turns']:
        turn_num = turn['turn_number']
        
        # Turn heading
        blocks.append(create_heading_block(f"Turn {turn_num}", level=2))
        
        # Messages in this turn
        for msg in turn['messages']:
            speaker = msg['speaker']
            text = msg['text']
            score = msg.get('score')  # Extract score
            
            if not text or text == "None":
                continue
            
            # Choose icon and color based on speaker
            if speaker == 'Mila':
                icon = "🤖"
                color = "blue_background"
            else:  # Tester
                icon = "🧪"
                color = "green_background"
            
            # Split long messages into multiple blocks
            chunks = split_text_for_notion(text, max_length=1900)
            
            # First chunk as callout
            blocks.append(create_callout_block(chunks[0], icon=icon, color=color))
            
            # Additional chunks as paragraphs
            for chunk in chunks[1:]:
                blocks.append(create_paragraph_block(chunk))
            
            # Add score indicator if present (only for Tester messages)
            if score is not None and speaker == 'Tester':
                # Determine color based on score
                if score >= 80:
                    score_icon = "🟢"
                    score_color = "green_background"
                    score_status = "Excellent"
                elif score >= 60:
                    score_icon = "🟡"
                    score_color = "yellow_background"
                    score_status = "Good"
                elif score >= 40:
                    score_icon = "🟠"
                    score_color = "orange_background"
                    score_status = "Fair"
                else:
                    score_icon = "🔴"
                    score_color = "red_background"
                    score_status = "Needs Work"
                
                # Add score callout
                score_text = f"{score_icon} Completeness Score: {score}/100 ({score_status})"
                blocks.append(create_callout_block(score_text, icon="📊", color=score_color))
        
        # Add divider after each turn (except the last one)
        if turn != parsed['turns'][-1]:
            blocks.append(create_divider_block())
    
    # If no turns were parsed, fall back to simple paragraph format
    if not blocks or len(blocks) <= 2:  # Only metadata and divider
        logger.warning("Could not parse conversation structure, using simple format")
        print("⚠️ Could not parse conversation structure, using simple format...")
        blocks = []
        paragraphs = conversation_text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            chunks = split_text_for_notion(para, max_length=1800)
            for chunk in chunks:
                blocks.append(create_paragraph_block(chunk))
    
    # Ensure we have at least one block
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