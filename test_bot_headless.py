# test_bot_headless.py
"""
Headless testing bot that replaces the Streamlit app.
Can be run via HTTP server for Playwright to interact with.

IMPROVEMENTS:
- Added timeout to OpenAI calls
- Session cleanup after inactivity
- Health check endpoint
- Better error handling and recovery
"""
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timedelta

from dotenv import load_dotenv
from logging_config import setup_logging, log_exception

# Load environment
load_dotenv()

# Set up logging
logger = setup_logging("test_bot_headless")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
MAX_TURNS = os.getenv("MAX_TURNS", "10")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))  # ✅ NEW: Configurable timeout
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))  # ✅ NEW: Session cleanup

logger.info("✓ Test Bot Headless Service Starting")
logger.info(f"OpenAI Model: {OPENAI_MODEL}")
logger.info(f"OpenAI API Key: {'✓ Set' if OPENAI_API_KEY else '✗ Missing'}")
logger.info(f"MAX_TURNS: {MAX_TURNS}")
logger.info(f"OpenAI Timeout: {OPENAI_TIMEOUT}s")
logger.info(f"Session Timeout: {SESSION_TIMEOUT_MINUTES} minutes")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("✓ OpenAI package available")
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI package not available")

# Initialize OpenAI client
openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✓ OpenAI client initialized")


class TestBotSession:
    """Manages a single test session"""
    
    def __init__(self, test_case: str, persona: str, test_case_details: str, test_case_prompt: str):
        self.test_case = test_case
        self.persona = persona
        self.test_case_details = test_case_details
        self.test_case_prompt = test_case_prompt
        self.chat_history: List[Dict[str, str]] = []
        self.score_history: List[int] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()  # ✅ NEW: Track last activity
        self.error_count = 0  # ✅ NEW: Track consecutive errors
        
        logger.info(f"Created new session: {test_case} / {persona}")
        logger.debug(f"  Details length: {len(test_case_details)} chars")
        logger.debug(f"  Prompt length: {len(test_case_prompt)} chars")
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if session has expired due to inactivity"""
        age = datetime.now() - self.last_activity
        return age > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    def build_system_prompt(self) -> str:
        """Build system prompt with placeholders replaced"""
        logger.debug("Building system prompt")
        
        if not self.test_case_prompt:
            logger.error("Test case prompt is required but missing")
            raise RuntimeError("Test case prompt is required")
        
        # Get MAX_TURNS from environment
        max_turns = os.getenv("MAX_TURNS", "10")
        
        prompt = (
            self.test_case_prompt
            .replace("{{test_case}}", self.test_case)
            .replace("{{persona}}", self.persona)
            .replace("{{test_case_details}}", self.test_case_details)
            .replace("{{max_turns}}", max_turns)
        )
        
        logger.debug(f"System prompt length: {len(prompt)} chars")
        logger.debug(f"MAX_TURNS: {max_turns}")
        return prompt
    
    def parse_json_response(self, reply: str) -> Tuple[str, Optional[int], bool]:
        """
        Parse JSON response from AI.
        
        Returns: (message, score, should_continue)
        """
        logger.debug(f"Parsing JSON response ({len(reply)} chars)")
        
        try:
            clean_reply = reply.strip()
            if clean_reply.startswith("```"):
                lines = clean_reply.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean_reply = "\n".join(lines).strip()
            
            data = json.loads(clean_reply)
            message = data.get("message", "")
            score = data.get("completeness_score")
            should_continue = data.get("should_continue", True)
            
            if score is not None:
                score = int(score)
                score = max(1, min(100, score))
            
            logger.debug(f"Parsed: message={len(message)} chars, score={score}, should_continue={should_continue}")
            self.error_count = 0  # ✅ Reset error count on success
            return message, score, should_continue
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"JSON parse failed, returning raw reply: {e}")
            self.error_count += 1
            return reply, None, True
    
    def get_response(self, user_message: str) -> Tuple[str, Optional[int], bool]:
        """
        Get AI response for user message.
        
        Returns: (reply, score, should_continue)
        """
        turn_number = len(self.chat_history) + 1
        logger.info(f"Getting AI response (turn {turn_number})")
        logger.debug(f"User message: {user_message[:100]}...")
        
        self.update_activity()  # ✅ Update activity timestamp
        
        if openai_client is None:
            logger.error("OPENAI_API_KEY not configured")
            return "❌ OPENAI_API_KEY not configured", None, False
        
        if not OPENAI_MODEL:
            logger.error("OPENAI_MODEL not set")
            return "❌ OPENAI_MODEL not set", None, False
        
        # ✅ Check for too many consecutive errors
        if self.error_count >= 3:
            logger.error("Too many consecutive errors, stopping session")
            return "❌ Too many consecutive errors. Please restart the session.", None, False
        
        try:
            system_prompt = self.build_system_prompt()
        except Exception as e:
            log_exception(logger, e, "build_system_prompt")
            self.error_count += 1
            return f"❌ System prompt error: {e}", None, False
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        for turn in self.chat_history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        messages.append({"role": "user", "content": user_message})
        
        logger.debug(f"Calling OpenAI with {len(messages)} messages (timeout: {OPENAI_TIMEOUT}s)")
        
        try:
            start_time = time.time()
            
            # ✅ ADD TIMEOUT TO OPENAI CALL
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                timeout=OPENAI_TIMEOUT,  # ✅ NEW: Add explicit timeout
            )
            
            duration = time.time() - start_time
            logger.debug(f"OpenAI response received in {duration:.2f}s")
            
            reply = resp.choices[0].message.content
            message, score, should_continue = self.parse_json_response(reply)
            
            # Save to history
            self.chat_history.append({
                "user": user_message,
                "assistant": message
            })
            
            if score is not None:
                self.score_history.append(score)
                logger.info(f"✓ Response received with score: {score}/100, should_continue: {should_continue}")
            else:
                logger.info(f"✓ Response received (no score), should_continue: {should_continue}")
            
            logger.debug(f"Message preview: {message[:100]}...")
            return message, score, should_continue
        
        except Exception as e:
            log_exception(logger, e, "OpenAI API call")
            logger.error(f"❌ OpenAI error: {e}")
            self.error_count += 1
            
            # ✅ Return different messages based on error type
            error_str = str(e).lower()
            if "timeout" in error_str:
                return f"❌ OpenAI timeout after {OPENAI_TIMEOUT}s. Please try again.", None, True
            elif "rate limit" in error_str:
                return f"❌ OpenAI rate limit exceeded. Please wait a moment and try again.", None, True
            else:
                return f"❌ OpenAI error: {e}", None, False


# Global session storage
active_sessions: Dict[str, TestBotSession] = {}


# ✅ NEW: Session cleanup function
def cleanup_expired_sessions():
    """Remove expired sessions from memory"""
    expired = [
        sid for sid, session in active_sessions.items()
        if session.is_expired()
    ]
    
    for sid in expired:
        logger.info(f"Cleaning up expired session: {sid}")
        del active_sessions[sid]
    
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions. Active: {len(active_sessions)}")


class TestBotHandler(BaseHTTPRequestHandler):
    """HTTP handler for test bot API"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        logger.debug(f"GET {parsed.path}")
        
        # ✅ Run cleanup on every request (lightweight operation)
        cleanup_expired_sessions()
        
        if parsed.path == "/health":
            logger.debug("Health check OK")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "active_sessions": len(active_sessions),
                "openai_available": OPENAI_AVAILABLE,
                "openai_model": OPENAI_MODEL
            }).encode())
        
        elif parsed.path == "/sessions":
            # ✅ NEW: List all active sessions
            logger.debug("Listing active sessions")
            sessions_info = {
                sid: {
                    "test_case": session.test_case,
                    "persona": session.persona,
                    "turns": len(session.chat_history),
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "age_minutes": (datetime.now() - session.created_at).total_seconds() / 60
                }
                for sid, session in active_sessions.items()
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "count": len(active_sessions),
                "sessions": sessions_info
            }, indent=2).encode())
        
        elif parsed.path == "/":
            # Serve a simple HTML interface for testing
            logger.debug("Serving root page")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Test Bot - Headless</title></head>
            <body>
                <h1>🧪 Test Bot - Headless Mode</h1>
                <p>This is the headless testing bot API.</p>
                <p><strong>Status:</strong> ✅ Running</p>
                <p><strong>Active Sessions:</strong> {len(active_sessions)}</p>
                <p><strong>OpenAI Model:</strong> {OPENAI_MODEL}</p>
                <p><strong>OpenAI Timeout:</strong> {OPENAI_TIMEOUT}s</p>
                <p><strong>Session Timeout:</strong> {SESSION_TIMEOUT_MINUTES} minutes</p>
                <h2>Endpoints:</h2>
                <ul>
                    <li>GET /health - Health check</li>
                    <li>GET /sessions - List active sessions</li>
                    <li>POST /session/create - Create new session</li>
                    <li>POST /session/{{id}}/message - Send message</li>
                    <li>GET /session/{{id}}/history - Get chat history</li>
                    <li>DELETE /session/{{id}} - Delete session</li>
                </ul>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        elif parsed.path.startswith("/session/") and parsed.path.endswith("/history"):
            # Get session history
            session_id = parsed.path.split("/")[2]
            logger.info(f"Getting history for session: {session_id}")
            
            if session_id not in active_sessions:
                logger.warning(f"Session not found: {session_id}")
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Session not found"}).encode())
                return
            
            session = active_sessions[session_id]
            session.update_activity()
            logger.debug(f"Returning {len(session.chat_history)} history entries")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "history": session.chat_history,
                "scores": session.score_history
            }).encode())
        
        else:
            logger.debug(f"Path not found: {parsed.path}")
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        logger.debug(f"POST {parsed.path}")
        
        # ✅ Run cleanup on every request
        cleanup_expired_sessions()
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in POST body: {e}")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return
        
        if parsed.path == "/session/create":
            # Create new session
            test_case = data.get("test_case", "")
            persona = data.get("persona", "")
            test_case_details = data.get("test_case_details", "")
            test_case_prompt = data.get("test_case_prompt", "")
            
            logger.info("Creating new session")
            logger.debug(f"  Test case: {test_case}")
            logger.debug(f"  Persona: {persona}")
            
            if not test_case or not persona:
                logger.error("Missing required fields for session creation")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Missing required fields: test_case, persona, test_case_prompt"
                }).encode())
                return
            
            # Generate session ID with timestamp to ensure uniqueness
            session_id = f"session_{int(time.time())}_{len(active_sessions)}"
            
            # Create session
            session = TestBotSession(
                test_case=test_case,
                persona=persona,
                test_case_details=test_case_details,
                test_case_prompt=test_case_prompt
            )
            active_sessions[session_id] = session
            
            logger.info(f"✓ Created session: {session_id}")
            logger.debug(f"  Total active sessions: {len(active_sessions)}")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "session_id": session_id,
                "message": "Session created"
            }).encode())
        
        elif parsed.path.startswith("/session/") and "/message" in parsed.path:
            # Send message to session
            session_id = parsed.path.split("/")[2]
            logger.info(f"Processing message for session: {session_id}")
            
            if session_id not in active_sessions:
                logger.warning(f"Session not found: {session_id}")
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Session not found"}).encode())
                return
            
            user_message = data.get("message", "")
            if not user_message:
                logger.error("Message is required but missing")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Message is required"}).encode())
                return
            
            session = active_sessions[session_id]
            reply, score, should_continue = session.get_response(user_message)
            
            logger.info(f"✓ Message processed (turn {len(session.chat_history)})")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "reply": reply,
                "score": score,
                "should_continue": should_continue,
                "turn": len(session.chat_history),
                "error_count": session.error_count
            }).encode())
        
        else:
            logger.debug(f"Path not found: {parsed.path}")
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def do_DELETE(self):
        """Handle DELETE requests"""
        parsed = urlparse(self.path)
        logger.debug(f"DELETE {parsed.path}")
        
        if parsed.path.startswith("/session/"):
            # Delete a specific session
            parts = parsed.path.split("/")
            if len(parts) >= 3:
                session_id = parts[2]
                logger.info(f"Deleting session: {session_id}")
                
                if session_id in active_sessions:
                    del active_sessions[session_id]
                    logger.info(f"✓ Deleted session: {session_id}")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "message": f"Session {session_id} deleted",
                        "active_sessions": len(active_sessions)
                    }).encode())
                else:
                    logger.warning(f"Session not found: {session_id}")
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Session not found"}).encode())
            else:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Invalid path")
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def log_message(self, format, *args):
        """Override to control logging"""
        # Only log non-200 responses to avoid spam
        if args[1] != '200':
            logger.warning(f"HTTP {args[1]}: {format % args}")


def run_server(port: int = 8501):
    """Run the test bot HTTP server"""
    logger.info("=" * 80)
    logger.info("🧪 Test Bot Server Starting")
    logger.info("=" * 80)
    logger.info(f"Port: {port}")
    logger.info(f"OpenAI Model: {OPENAI_MODEL}")
    logger.info(f"OpenAI API Key: {'✓ Set' if OPENAI_API_KEY else '✗ Missing'}")
    logger.info(f"OpenAI Available: {OPENAI_AVAILABLE}")
    logger.info(f"OpenAI Timeout: {OPENAI_TIMEOUT}s")
    logger.info(f"MAX_TURNS: {MAX_TURNS}")
    logger.info(f"Session Timeout: {SESSION_TIMEOUT_MINUTES} minutes")
    logger.info("=" * 80)
    
    print(f"🧪 Test Bot Server running on http://0.0.0.0:{port}")
    print(f"   OpenAI Model: {OPENAI_MODEL}")
    print(f"   OpenAI API Key: {'✓ Set' if OPENAI_API_KEY else '✗ Missing'}")
    print(f"   OpenAI Timeout: {OPENAI_TIMEOUT}s")
    print(f"   MAX_TURNS: {MAX_TURNS}")
    print(f"   Session Timeout: {SESSION_TIMEOUT_MINUTES} minutes")
    
    server = HTTPServer(('0.0.0.0', port), TestBotHandler)
    
    try:
        logger.info("Server starting to serve requests")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down server...")
        print("\n👋 Shutting down...")
    except Exception as e:
        log_exception(logger, e, "run_server")
        raise
    finally:
        logger.info("Server stopped")


if __name__ == "__main__":
    import sys
    
    logger.info("Script started from command line")
    logger.info(f"Arguments: {sys.argv}")
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    logger.info(f"Using port: {port}")
    
    try:
        run_server(port)
    except Exception as e:
        log_exception(logger, e, "main")
        raise