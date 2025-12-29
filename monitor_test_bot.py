#!/usr/bin/env python3
"""
monitor_test_bot.py
Monitor the test bot API for health and stuck sessions.
Usage: python monitor_test_bot.py [--interval SECONDS] [--alert-age MINUTES]
"""
import os
import sys
import time
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

TESTER_API_URL = os.getenv("TESTER_API_URL", "http://localhost:8501")
CHECK_INTERVAL = 30  # seconds between checks
STUCK_SESSION_THRESHOLD = 15  # minutes - alert if session inactive for this long


def check_health():
    """Check test bot health"""
    try:
        response = httpx.get(f"{TESTER_API_URL}/health", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Test Bot Health: OK")
        print(f"   Status: {data.get('status')}")
        print(f"   Active Sessions: {data.get('active_sessions')}")
        print(f"   OpenAI Available: {data.get('openai_available')}")
        print(f"   OpenAI Model: {data.get('openai_model')}")
        
        return True, data
        
    except Exception as e:
        print(f"❌ Test Bot Health: FAILED - {e}")
        return False, None


def check_sessions():
    """Check for stuck or long-running sessions"""
    try:
        response = httpx.get(f"{TESTER_API_URL}/sessions", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        sessions = data.get('sessions', {})
        count = data.get('count', 0)
        
        if count == 0:
            print(f"📊 Sessions: 0 active")
            return
        
        print(f"\n📊 Sessions: {count} active")
        
        now = datetime.now()
        stuck_sessions = []
        
        for sid, info in sessions.items():
            test_case = info.get('test_case', 'Unknown')
            turns = info.get('turns', 0)
            age_minutes = info.get('age_minutes', 0)
            
            # Parse last activity
            last_activity_str = info.get('last_activity', '')
            try:
                last_activity = datetime.fromisoformat(last_activity_str)
                inactive_minutes = (now - last_activity).total_seconds() / 60
            except:
                inactive_minutes = age_minutes
            
            # Detect stuck sessions
            status = "🟢 OK"
            if inactive_minutes > STUCK_SESSION_THRESHOLD:
                status = "🔴 STUCK"
                stuck_sessions.append((sid, test_case, inactive_minutes))
            elif age_minutes > 30:
                status = "🟡 OLD"
            
            print(f"   {status} {sid}")
            print(f"      Test: {test_case[:50]}...")
            print(f"      Turns: {turns}")
            print(f"      Age: {age_minutes:.1f} min")
            print(f"      Inactive: {inactive_minutes:.1f} min")
        
        # Alert on stuck sessions
        if stuck_sessions:
            print(f"\n⚠️  ALERT: {len(stuck_sessions)} stuck session(s) detected!")
            for sid, test_case, inactive_min in stuck_sessions:
                print(f"   • {sid} ({test_case[:30]}...) - inactive for {inactive_min:.1f} min")
                print(f"     Consider running: curl -X DELETE {TESTER_API_URL}/session/{sid}")
        
    except Exception as e:
        print(f"❌ Error checking sessions: {e}")


def main():
    """Main monitoring loop"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Test Bot API')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL,
                       help=f'Check interval in seconds (default: {CHECK_INTERVAL})')
    parser.add_argument('--alert-age', type=int, default=STUCK_SESSION_THRESHOLD,
                       help=f'Alert if session inactive for this many minutes (default: {STUCK_SESSION_THRESHOLD})')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    global CHECK_INTERVAL, STUCK_SESSION_THRESHOLD
    CHECK_INTERVAL = args.interval
    STUCK_SESSION_THRESHOLD = args.alert_age
    
    print(f"🔍 Test Bot Monitor")
    print(f"   URL: {TESTER_API_URL}")
    print(f"   Check Interval: {CHECK_INTERVAL}s")
    print(f"   Stuck Threshold: {STUCK_SESSION_THRESHOLD} min")
    print("=" * 60)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n[{timestamp}] Check #{iteration}")
            print("-" * 60)
            
            # Check health
            healthy, health_data = check_health()
            
            # Check sessions
            if healthy:
                check_sessions()
            
            if args.once:
                break
            
            # Wait for next check
            print(f"\n⏳ Next check in {CHECK_INTERVAL}s...")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()