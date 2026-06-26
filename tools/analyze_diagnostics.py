#!/usr/bin/env python3
"""
analyze_diagnostics.py
Analyze diagnostic data to identify patterns, errors, and edge cases
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


class DiagnosticAnalyzer:
    """Analyze diagnostic data from test runs"""
    
    def __init__(self, diagnostics_dir: str = "diagnostics"):
        self.diagnostics_dir = Path(diagnostics_dir)
        self.sessions = []
        self.load_sessions()
    
    def load_sessions(self):
        """Load all diagnostic sessions"""
        if not self.diagnostics_dir.exists():
            print(f"❌ Diagnostics directory not found: {self.diagnostics_dir}")
            return
        
        # Group files by session ID
        session_files = defaultdict(dict)
        
        for file in self.diagnostics_dir.glob("*"):
            if file.is_file():
                parts = file.stem.split('_', 2)
                if len(parts) >= 2:
                    session_id = f"{parts[0]}_{parts[1]}"
                    file_type = '_'.join(parts[2:]) if len(parts) > 2 else file.suffix
                    session_files[session_id][file_type] = file
        
        # Load data for each session
        for session_id, files in session_files.items():
            session = {
                "id": session_id,
                "files": files,
                "console_logs": [],
                "network": [],
                "errors": [],
                "ui_states": [],
                "dom_snapshots": []
            }
            
            # Load JSON files
            for file_type, filepath in files.items():
                if filepath.suffix == '.json':
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            if 'console' in file_type:
                                session['console_logs'] = data
                            elif 'network' in file_type:
                                session['network'] = data
                            elif 'errors' in file_type:
                                session['errors'] = data
                            elif 'ui_states' in file_type:
                                session['ui_states'] = data
                            elif 'dom_snapshots' in file_type:
                                session['dom_snapshots'] = data
                    except Exception as e:
                        print(f"Warning: Could not load {filepath}: {e}")
            
            self.sessions.append(session)
        
        print(f"✓ Loaded {len(self.sessions)} diagnostic sessions")
    
    def analyze_all(self):
        """Run all analyses"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        print("=" * 80 + "\n")
        
        self.analyze_errors()
        self.analyze_ui_patterns()
        self.analyze_network_issues()
        self.analyze_chat_states()
        self.identify_failure_patterns()
        self.generate_recommendations()
    
    def analyze_errors(self):
        """Analyze all errors across sessions"""
        print("\n" + "=" * 80)
        print("ERROR ANALYSIS")
        print("=" * 80)
        
        all_errors = []
        for session in self.sessions:
            all_errors.extend(session.get('errors', []))
        
        if not all_errors:
            print("✓ No JavaScript errors found")
            return
        
        print(f"\n📊 Found {len(all_errors)} JavaScript errors across {len(self.sessions)} sessions")
        
        # Group by error message
        error_counts = Counter(err.get('message', 'Unknown')[:100] for err in all_errors)
        
        print("\nMost common errors:")
        for error, count in error_counts.most_common(10):
            print(f"  • {count}x: {error}")
    
    def analyze_ui_patterns(self):
        """Analyze UI state patterns"""
        print("\n" + "=" * 80)
        print("UI PATTERN ANALYSIS")
        print("=" * 80)
        
        # Collect all selectors that worked
        working_selectors = defaultdict(int)
        error_selectors = defaultdict(int)
        
        for session in self.sessions:
            for ui_state in session.get('ui_states', []):
                # Message bubble selectors
                for selector, info in ui_state.get('message_bubbles', {}).items():
                    if info.get('count', 0) > 0:
                        working_selectors[f"message: {selector}"] += 1
                
                # Error indicators
                for selector, info in ui_state.get('error_indicators', {}).items():
                    if info.get('visible', False):
                        error_selectors[f"error: {selector}"] += 1
        
        print("\n📍 Working message bubble selectors:")
        for selector, count in sorted(working_selectors.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  • {selector}: found in {count} states")
        
        if error_selectors:
            print("\n🚨 Error indicators detected:")
            for selector, count in sorted(error_selectors.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {selector}: appeared {count} times")
    
    def analyze_network_issues(self):
        """Analyze network request/response patterns"""
        print("\n" + "=" * 80)
        print("NETWORK ANALYSIS")
        print("=" * 80)
        
        all_requests = []
        all_responses = []
        
        for session in self.sessions:
            for event in session.get('network', []):
                if event.get('type') == 'request':
                    all_requests.append(event)
                elif event.get('type') == 'response':
                    all_responses.append(event)
        
        print(f"\n📡 Total network activity:")
        print(f"  Requests: {len(all_requests)}")
        print(f"  Responses: {len(all_responses)}")
        
        # Analyze status codes
        status_codes = Counter(resp.get('status', 0) for resp in all_responses)
        
        print("\n📊 Response status codes:")
        for status, count in sorted(status_codes.items()):
            emoji = "✓" if 200 <= status < 300 else "⚠️" if 300 <= status < 400 else "❌"
            print(f"  {emoji} {status}: {count}")
        
        # Find failed requests
        failed = [resp for resp in all_responses if resp.get('status', 0) >= 400]
        if failed:
            print(f"\n❌ Failed requests ({len(failed)}):")
            for resp in failed[:5]:
                print(f"  • {resp.get('status')} {resp.get('url', '')[:80]}")
    
    def analyze_chat_states(self):
        """Analyze chat state progression"""
        print("\n" + "=" * 80)
        print("CHAT STATE ANALYSIS")
        print("=" * 80)
        
        for session in self.sessions:
            ui_states = session.get('ui_states', [])
            if not ui_states:
                continue
            
            print(f"\n📝 Session {session['id']}:")
            print(f"  State snapshots: {len(ui_states)}")
            
            # Analyze progression
            for i, state in enumerate(ui_states):
                label = state.get('label', f'state_{i}')
                errors = len([v for v in state.get('error_indicators', {}).values() if v.get('visible')])
                messages = sum(v.get('count', 0) for v in state.get('message_bubbles', {}).values())
                loading = sum(v.get('count', 0) for v in state.get('loading_indicators', {}).values())
                
                status = "✓" if errors == 0 else "❌"
                print(f"    {status} {label}: {messages} msgs, {loading} loading, {errors} errors")
    
    def identify_failure_patterns(self):
        """Identify common failure patterns"""
        print("\n" + "=" * 80)
        print("FAILURE PATTERN IDENTIFICATION")
        print("=" * 80)
        
        patterns = {
            "backend_errors": 0,
            "timeout_errors": 0,
            "network_failures": 0,
            "ui_errors": 0,
            "message_not_appearing": 0
        }
        
        for session in self.sessions:
            # Check for backend errors
            for ui_state in session.get('ui_states', []):
                for selector, info in ui_state.get('error_indicators', {}).items():
                    if 'terribly wrong' in info.get('text', '').lower():
                        patterns["backend_errors"] += 1
                    if 'retry' in selector.lower():
                        patterns["backend_errors"] += 1
            
            # Check for network failures
            for event in session.get('network', []):
                if event.get('type') == 'response' and event.get('status', 0) >= 500:
                    patterns["network_failures"] += 1
        
        print("\n🔍 Detected patterns:")
        for pattern, count in patterns.items():
            if count > 0:
                emoji = "🚨" if count > 5 else "⚠️" if count > 0 else "✓"
                print(f"  {emoji} {pattern.replace('_', ' ').title()}: {count}")
    
    def generate_recommendations(self):
        """Generate recommendations for improving automation"""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR 100% RELIABLE AUTOMATION")
        print("=" * 80)
        
        recommendations = []
        
        # Analyze what we learned
        has_backend_errors = any(
            'terribly wrong' in str(state.get('error_indicators', {}))
            for session in self.sessions
            for state in session.get('ui_states', [])
        )
        
        has_retry_buttons = any(
            'retry' in str(state.get('buttons', {})).lower()
            for session in self.sessions
            for state in session.get('ui_states', [])
        )
        
        # Generate recommendations
        if has_backend_errors:
            recommendations.append({
                "priority": "HIGH",
                "issue": "Backend errors detected ('something went terribly wrong')",
                "recommendation": "Add error detection for backend failures",
                "implementation": """
# After sending message, check for error banner
try:
    error_locator = page.locator("text='something went terribly wrong'").first
    if error_locator.is_visible(timeout=2000):
        logger.error("Backend error detected")
        return {"status": "error", "type": "backend_crash"}
except:
    pass
"""
            })
        
        if has_retry_buttons:
            recommendations.append({
                "priority": "HIGH",
                "issue": "Retry buttons appearing (indicates failed operations)",
                "recommendation": "Detect and optionally click retry buttons",
                "implementation": """
# Check for retry button
try:
    retry_btn = page.locator("button:has-text('Retry last instruction')").first
    if retry_btn.is_visible(timeout=2000):
        logger.warning("Retry button detected - operation failed")
        # Option 1: Click retry
        retry_btn.click()
        # Option 2: Mark as failure and continue
        return {"status": "retry_needed", "action": "click_retry"}
except:
    pass
"""
            })
        
        # Always recommend multiple selector fallbacks
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "Single selector dependencies can fail",
            "recommendation": "Use multiple fallback selectors",
            "implementation": """
# Try multiple selectors in order
message_selectors = [
    ".message-bubble.ai-message.ai-message-text",
    ".ai-message",
    "[class*='message'][class*='ai']",
    "[role='listitem']"
]

for selector in message_selectors:
    try:
        count = page.locator(selector).count()
        if count > 0:
            logger.info(f"Using selector: {selector} ({count} messages)")
            return selector
    except:
        continue

logger.error("No message selector worked")
"""
        })
        
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "Message state changes not fully captured",
            "recommendation": "Monitor multiple state indicators simultaneously",
            "implementation": """
# Check multiple state indicators at once
def check_message_state(page):
    state = {
        "new_message": False,
        "error": False,
        "loading": False,
        "retry_available": False
    }
    
    # Check for new message
    try:
        if page.locator(".message-bubble").count() > previous_count:
            state["new_message"] = True
    except: pass
    
    # Check for error
    try:
        if page.locator("text='terribly wrong'").is_visible(timeout=500):
            state["error"] = True
    except: pass
    
    # Check for loading
    try:
        if page.locator("text='...'").is_visible(timeout=500):
            state["loading"] = True
    except: pass
    
    # Check for retry
    try:
        if page.locator("button:has-text('Retry')").is_visible(timeout=500):
            state["retry_available"] = True
    except: pass
    
    return state
"""
        })
        
        recommendations.append({
            "priority": "LOW",
            "issue": "Network timing issues",
            "recommendation": "Add network idle detection",
            "implementation": """
# Wait for network to be idle
try:
    page.wait_for_load_state("networkidle", timeout=10000)
    logger.info("Network is idle")
except:
    logger.warning("Network idle timeout")
"""
        })
        
        # Print recommendations
        print("\n🎯 Recommended improvements:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['priority']}] {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Implementation:{rec['implementation']}")
            print()
    
    def export_selector_library(self, output_file: str = "selector_library.json"):
        """Export all working selectors as a reference library"""
        library = {
            "generated": datetime.now().isoformat(),
            "sessions_analyzed": len(self.sessions),
            "selectors": {
                "message_bubbles": {},
                "input_fields": {},
                "buttons": {},
                "error_indicators": {},
                "loading_indicators": {}
            }
        }
        
        # Collect all selectors with their success rates
        for session in self.sessions:
            for ui_state in session.get('ui_states', []):
                for category in library["selectors"].keys():
                    for selector, info in ui_state.get(category, {}).items():
                        if selector not in library["selectors"][category]:
                            library["selectors"][category][selector] = {
                                "found_count": 0,
                                "visible_count": 0,
                                "enabled_count": 0
                            }
                        
                        if info.get('count', 0) > 0:
                            library["selectors"][category][selector]["found_count"] += 1
                        if info.get('visible', False):
                            library["selectors"][category][selector]["visible_count"] += 1
                        if info.get('enabled', False):
                            library["selectors"][category][selector]["enabled_count"] += 1
        
        # Save library
        filepath = self.diagnostics_dir / output_file
        with open(filepath, 'w') as f:
            json.dump(library, f, indent=2)
        
        print(f"\n✓ Selector library exported to {filepath}")
        return library


def main():
    parser = argparse.ArgumentParser(description="Analyze diagnostic data from Playwright automation")
    parser.add_argument("--dir", default="diagnostics", help="Diagnostics directory")
    parser.add_argument("--export-selectors", action="store_true", help="Export selector library")
    parser.add_argument("--session", help="Analyze specific session ID")
    
    args = parser.parse_args()
    
    analyzer = DiagnosticAnalyzer(args.dir)
    
    if not analyzer.sessions:
        print("❌ No diagnostic sessions found")
        return
    
    if args.session:
        # Analyze specific session
        session = next((s for s in analyzer.sessions if s['id'] == args.session), None)
        if session:
            print(f"Analyzing session: {args.session}")
            # TODO: Add detailed single-session analysis
        else:
            print(f"❌ Session not found: {args.session}")
    else:
        # Analyze all sessions
        analyzer.analyze_all()
    
    if args.export_selectors:
        analyzer.export_selector_library()


if __name__ == "__main__":
    main()