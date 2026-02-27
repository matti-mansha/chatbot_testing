"""
diagnostic_utils.py
Comprehensive diagnostic utilities for Playwright automation
Captures HTML, JavaScript errors, network activity, and all UI states
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from playwright.sync_api import Page, ConsoleMessage, Request, Response


class DiagnosticCapture:
    """Captures comprehensive diagnostic information during automation"""
    
    def __init__(self, output_dir: str = "diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for captured data
        self.console_logs: List[Dict] = []
        self.network_requests: List[Dict] = []
        self.dom_snapshots: List[Dict] = []
        self.errors: List[Dict] = []
        self.ui_states: List[Dict] = []
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def setup_listeners(self, page: Page):
        """Set up all diagnostic listeners on the page"""
        
        # Console message listener
        page.on("console", self._on_console_message)
        
        # JavaScript error listener
        page.on("pageerror", self._on_page_error)
        
        # Network request listener
        page.on("request", self._on_request)
        
        # Network response listener
        page.on("response", self._on_response)
        
        print(f"✓ Diagnostic listeners activated for session: {self.session_id}")
    
    def _on_console_message(self, msg: ConsoleMessage):
        """Capture console messages"""
        self.console_logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": msg.type,
            "text": msg.text,
            "location": f"{msg.location.get('url', '')}:{msg.location.get('lineNumber', '')}"
        })
    
    def _on_page_error(self, error):
        """Capture JavaScript errors"""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "type": "javascript_error",
            "message": str(error)
        })
    
    def _on_request(self, request: Request):
        """Capture network requests"""
        self.network_requests.append({
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "method": request.method,
            "url": request.url,
            "resource_type": request.resource_type,
            "headers": dict(request.headers)
        })
    
    def _on_response(self, response: Response):
        """Capture network responses"""
        try:
            self.network_requests.append({
                "timestamp": datetime.now().isoformat(),
                "type": "response",
                "url": response.url,
                "status": response.status,
                "status_text": response.status_text,
                "headers": dict(response.headers)
            })
        except Exception as e:
            print(f"Error capturing response: {e}")
    
    def capture_dom_snapshot(self, page: Page, label: str):
        """Capture complete DOM snapshot"""
        try:
            html = page.content()
            
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "url": page.url,
                "title": page.title(),
                "html_length": len(html)
            }
            
            # Save HTML to file
            filename = f"{self.session_id}_{label}_dom.html"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            snapshot["html_file"] = str(filepath)
            self.dom_snapshots.append(snapshot)
            
            print(f"✓ DOM snapshot saved: {filename}")
            
        except Exception as e:
            print(f"Error capturing DOM: {e}")
    
    def detect_all_chat_elements(self, page: Page, label: str) -> Dict:
        """Detect all possible chat-related elements"""
        
        elements = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "message_bubbles": {},
            "input_fields": {},
            "buttons": {},
            "error_indicators": {},
            "loading_indicators": {},
            "chat_state": {}
        }
        
        # Message bubble selectors to try
        message_selectors = [
            ".message-bubble",
            ".ai-message",
            ".user-message",
            "[class*='message']",
            "[class*='bubble']",
            "[role='log']",
            "[role='listitem']",
            ".ai-deepchat",
        ]
        
        for selector in message_selectors:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    elements["message_bubbles"][selector] = {
                        "count": count,
                        "visible": page.locator(selector).first.is_visible() if count > 0 else False
                    }
            except:
                pass
        
        # Input field selectors
        input_selectors = [
            "input[type='text']",
            "textarea",
            "[contenteditable='true']",
            "[placeholder*='Ask']",
            "[placeholder*='message']",
            ".input",
            "[role='textbox']",
        ]
        
        for selector in input_selectors:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    elements["input_fields"][selector] = {
                        "count": count,
                        "visible": page.locator(selector).first.is_visible() if count > 0 else False,
                        "enabled": page.locator(selector).first.is_enabled() if count > 0 else False
                    }
            except:
                pass
        
        # Button selectors
        button_selectors = [
            "button[type='submit']",
            "button:has-text('Send')",
            "button:has-text('Retry')",
            "[aria-label*='Send']",
            ".send-button",
            ".submit-button",
        ]
        
        for selector in button_selectors:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    elements["buttons"][selector] = {
                        "count": count,
                        "visible": page.locator(selector).first.is_visible() if count > 0 else False,
                        "enabled": page.locator(selector).first.is_enabled() if count > 0 else False
                    }
            except:
                pass
        
        # Error indicators
        error_selectors = [
            "[role='alert']",
            ".error",
            ".warning",
            "[class*='error']",
            "text='something went terribly wrong'",
            "text='error'",
            "text='failed'",
            "button:has-text('Retry')",
        ]
        
        for selector in error_selectors:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    elements["error_indicators"][selector] = {
                        "count": count,
                        "visible": page.locator(selector).first.is_visible() if count > 0 else False,
                        "text": page.locator(selector).first.text_content() if count > 0 else ""
                    }
            except:
                pass
        
        # Loading indicators
        loading_selectors = [
            "text='...'",
            "[class*='loading']",
            "[class*='spinner']",
            "[class*='typing']",
            ".dots",
        ]
        
        for selector in loading_selectors:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    elements["loading_indicators"][selector] = {
                        "count": count,
                        "visible": page.locator(selector).first.is_visible() if count > 0 else False
                    }
            except:
                pass
        
        # Chat state
        try:
            elements["chat_state"] = {
                "url": page.url,
                "title": page.title(),
                "viewport": page.viewport_size,
            }
        except:
            pass
        
        self.ui_states.append(elements)
        return elements
    
    def check_for_errors(self, page: Page) -> Dict[str, Any]:
        """Check for all types of errors on the page"""
        
        error_state = {
            "timestamp": datetime.now().isoformat(),
            "has_errors": False,
            "error_messages": [],
            "error_types": []
        }
        
        # Check for visible error messages
        error_patterns = [
            ("something went terribly wrong", "backend_error"),
            ("error occurred", "general_error"),
            ("failed to", "failure_error"),
            ("unable to", "unable_error"),
            ("try again", "retry_prompt"),
            ("Retry last instruction", "retry_button"),
        ]
        
        for pattern, error_type in error_patterns:
            try:
                locator = page.locator(f"text='{pattern}'").first
                if locator.is_visible(timeout=1000):
                    error_state["has_errors"] = True
                    error_state["error_types"].append(error_type)
                    
                    # Try to get full error text
                    try:
                        full_text = locator.text_content()
                        error_state["error_messages"].append(full_text)
                    except:
                        error_state["error_messages"].append(pattern)
            except:
                pass
        
        # Check for error styling/classes
        error_classes = [
            "[class*='error']",
            "[class*='alert']",
            "[class*='warning']",
            "[role='alert']",
        ]
        
        for selector in error_classes:
            try:
                locator = page.locator(selector).first
                if locator.is_visible(timeout=1000):
                    error_state["has_errors"] = True
                    try:
                        text = locator.text_content()
                        if text and text not in error_state["error_messages"]:
                            error_state["error_messages"].append(text)
                    except:
                        pass
            except:
                pass
        
        return error_state
    
    def save_diagnostics(self):
        """Save all diagnostic data to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save console logs
        if self.console_logs:
            filepath = self.output_dir / f"{self.session_id}_console_logs.json"
            with open(filepath, 'w') as f:
                json.dump(self.console_logs, f, indent=2)
            print(f"✓ Saved {len(self.console_logs)} console logs to {filepath}")
        
        # Save network requests
        if self.network_requests:
            filepath = self.output_dir / f"{self.session_id}_network.json"
            with open(filepath, 'w') as f:
                json.dump(self.network_requests, f, indent=2)
            print(f"✓ Saved {len(self.network_requests)} network events to {filepath}")
        
        # Save errors
        if self.errors:
            filepath = self.output_dir / f"{self.session_id}_errors.json"
            with open(filepath, 'w') as f:
                json.dump(self.errors, f, indent=2)
            print(f"✓ Saved {len(self.errors)} errors to {filepath}")
        
        # Save UI states
        if self.ui_states:
            filepath = self.output_dir / f"{self.session_id}_ui_states.json"
            with open(filepath, 'w') as f:
                json.dump(self.ui_states, f, indent=2)
            print(f"✓ Saved {len(self.ui_states)} UI state snapshots to {filepath}")
        
        # Save DOM snapshot metadata
        if self.dom_snapshots:
            filepath = self.output_dir / f"{self.session_id}_dom_snapshots.json"
            with open(filepath, 'w') as f:
                json.dump(self.dom_snapshots, f, indent=2)
            print(f"✓ Saved {len(self.dom_snapshots)} DOM snapshot records to {filepath}")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report"""
        
        report = []
        report.append("=" * 80)
        report.append("DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Session ID: {self.session_id}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        report.append("SUMMARY:")
        report.append(f"  Console Logs: {len(self.console_logs)}")
        report.append(f"  Network Events: {len(self.network_requests)}")
        report.append(f"  JavaScript Errors: {len(self.errors)}")
        report.append(f"  UI State Snapshots: {len(self.ui_states)}")
        report.append(f"  DOM Snapshots: {len(self.dom_snapshots)}")
        report.append("")
        
        # Console log breakdown
        if self.console_logs:
            report.append("CONSOLE LOGS BY TYPE:")
            log_types = {}
            for log in self.console_logs:
                log_type = log.get("type", "unknown")
                log_types[log_type] = log_types.get(log_type, 0) + 1
            for log_type, count in sorted(log_types.items()):
                report.append(f"  {log_type}: {count}")
            report.append("")
        
        # JavaScript errors
        if self.errors:
            report.append("JAVASCRIPT ERRORS:")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5
                report.append(f"  {i}. {error.get('message', '')[:100]}")
            if len(self.errors) > 5:
                report.append(f"  ... and {len(self.errors) - 5} more")
            report.append("")
        
        # Network requests
        if self.network_requests:
            report.append("NETWORK ACTIVITY:")
            request_count = sum(1 for r in self.network_requests if r.get("type") == "request")
            response_count = sum(1 for r in self.network_requests if r.get("type") == "response")
            report.append(f"  Requests: {request_count}")
            report.append(f"  Responses: {response_count}")
            
            # Status code breakdown
            status_codes = {}
            for event in self.network_requests:
                if event.get("type") == "response":
                    status = event.get("status", "unknown")
                    status_codes[status] = status_codes.get(status, 0) + 1
            if status_codes:
                report.append("  Status Codes:")
                for status, count in sorted(status_codes.items()):
                    report.append(f"    {status}: {count}")
            report.append("")
        
        # UI States
        if self.ui_states:
            report.append("UI STATE SNAPSHOTS:")
            for state in self.ui_states:
                label = state.get("label", "unknown")
                error_count = len(state.get("error_indicators", {}))
                message_count = len(state.get("message_bubbles", {}))
                report.append(f"  {label}:")
                report.append(f"    Message bubble selectors found: {message_count}")
                report.append(f"    Error indicators found: {error_count}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Save report
        filepath = self.output_dir / f"{self.session_id}_REPORT.txt"
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Diagnostic report saved to {filepath}\n")
        
        # Also print to console
        print('\n'.join(report))


# Convenience function for quick diagnostic runs
def run_with_diagnostics(page: Page, test_function, output_dir: str = "diagnostics"):
    """
    Run a test function with full diagnostics enabled
    
    Usage:
        def my_test(page):
            page.goto("https://example.com")
            # ... test code ...
        
        run_with_diagnostics(page, my_test)
    """
    
    diagnostics = DiagnosticCapture(output_dir)
    diagnostics.setup_listeners(page)
    
    try:
        # Run the test
        test_function(page, diagnostics)
    finally:
        # Always save diagnostics
        diagnostics.save_diagnostics()
    
    return diagnostics