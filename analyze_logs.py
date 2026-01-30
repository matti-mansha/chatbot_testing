#!/usr/bin/env python3
# analyze_logs.py
"""
Log Analysis Utility

Helps analyze and debug logs from the testing system.
Provides filtering, statistics, and error tracking.
"""
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict


class LogEntry:
    """Represents a single log entry"""
    
    def __init__(self, line: str):
        self.raw = line
        self.timestamp = None
        self.service = None
        self.level = None
        self.location = None
        self.message = None
        
        self.parse(line)
    
    def parse(self, line: str):
        """Parse a log line"""
        # Format: YYYY-MM-DD HH:MM:SS | service | LEVEL | file:line | message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+) \| (\w+)\s+ \| ([^|]+) \| (.+)'
        match = re.match(pattern, line)
        
        if match:
            self.timestamp = match.group(1)
            self.service = match.group(2)
            self.level = match.group(3).strip()
            self.location = match.group(4).strip()
            self.message = match.group(5).strip()
        else:
            # Fallback for lines that don't match format
            self.message = line
    
    def __str__(self):
        if self.timestamp:
            return f"[{self.timestamp}] {self.level:8s} | {self.message}"
        return self.message


class LogAnalyzer:
    """Analyzes log files"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.entries: List[LogEntry] = []
        self.stats: Dict[str, Any] = defaultdict(int)
        
    def load_logs(self, service: str = None, date: str = None):
        """Load logs from files"""
        if not self.log_dir.exists():
            print(f"❌ Log directory not found: {self.log_dir}")
            return
        
        # Find matching log files
        pattern = "*"
        if service:
            pattern = f"{service}_*"
        if date:
            pattern = f"*{date}*.log"
        
        log_files = list(self.log_dir.glob(pattern + ".log"))
        
        if not log_files:
            print(f"❌ No log files found matching: {pattern}")
            return
        
        print(f"📄 Loading {len(log_files)} log file(s)...")
        
        for log_file in log_files:
            print(f"   Loading: {log_file.name}")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = LogEntry(line)
                            self.entries.append(entry)
                            
                            # Update stats
                            if entry.level:
                                self.stats[f"level_{entry.level}"] += 1
                            if entry.service:
                                self.stats[f"service_{entry.service}"] += 1
            except Exception as e:
                print(f"   ⚠️  Error loading {log_file.name}: {e}")
        
        print(f"✓ Loaded {len(self.entries)} log entries\n")
    
    def filter_entries(
        self,
        level: str = None,
        service: str = None,
        search: str = None,
        contains_error: bool = False
    ) -> List[LogEntry]:
        """Filter log entries"""
        filtered = self.entries
        
        if level:
            filtered = [e for e in filtered if e.level and e.level.upper() == level.upper()]
        
        if service:
            filtered = [e for e in filtered if e.service and service.lower() in e.service.lower()]
        
        if search:
            search_lower = search.lower()
            filtered = [e for e in filtered if e.message and search_lower in e.message.lower()]
        
        if contains_error:
            filtered = [e for e in filtered if e.level in ['ERROR', 'CRITICAL', 'WARNING']]
        
        return filtered
    
    def show_stats(self):
        """Show log statistics"""
        print("=" * 80)
        print("LOG STATISTICS")
        print("=" * 80)
        print(f"Total entries: {len(self.entries)}\n")
        
        # By level
        print("By Level:")
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            count = self.stats.get(f"level_{level}", 0)
            if count > 0:
                bar = "█" * min(50, count // 10)
                print(f"  {level:8s}: {count:6d} {bar}")
        print()
        
        # By service
        print("By Service:")
        services = [k.replace("service_", "") for k in self.stats.keys() if k.startswith("service_")]
        for service in services:
            count = self.stats[f"service_{service}"]
            bar = "█" * min(50, count // 10)
            print(f"  {service:20s}: {count:6d} {bar}")
        print()
    
    def show_errors(self, limit: int = 50):
        """Show recent errors"""
        errors = self.filter_entries(contains_error=True)
        
        print("=" * 80)
        print(f"ERRORS AND WARNINGS (showing last {min(limit, len(errors))})")
        print("=" * 80)
        
        if not errors:
            print("✓ No errors found!\n")
            return
        
        # Show most recent
        for entry in errors[-limit:]:
            print(entry)
        print()
    
    def show_timeline(self, interval_minutes: int = 5):
        """Show activity timeline"""
        print("=" * 80)
        print(f"ACTIVITY TIMELINE (grouped by {interval_minutes} minutes)")
        print("=" * 80)
        
        timeline: Dict[str, int] = defaultdict(int)
        
        for entry in self.entries:
            if entry.timestamp:
                # Round to interval
                try:
                    dt = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")
                    rounded_min = (dt.minute // interval_minutes) * interval_minutes
                    key = dt.strftime(f"%Y-%m-%d %H:{rounded_min:02d}")
                    timeline[key] += 1
                except:
                    pass
        
        for time_slot in sorted(timeline.keys()):
            count = timeline[time_slot]
            bar = "█" * min(50, count // 10)
            print(f"{time_slot}: {count:6d} {bar}")
        print()
    
    def search(self, query: str, context: int = 2):
        """Search logs with context"""
        results = self.filter_entries(search=query)
        
        print("=" * 80)
        print(f"SEARCH RESULTS: '{query}' ({len(results)} matches)")
        print("=" * 80)
        
        if not results:
            print("No matches found.\n")
            return
        
        # Show with context
        for i, entry in enumerate(results[:50]):  # Limit to 50 matches
            # Find index in full entries
            try:
                idx = self.entries.index(entry)
                
                # Show context before
                for j in range(max(0, idx - context), idx):
                    print(f"  {self.entries[j]}")
                
                # Show match (highlighted)
                print(f">>> {entry}")
                
                # Show context after
                for j in range(idx + 1, min(len(self.entries), idx + context + 1)):
                    print(f"  {self.entries[j]}")
                
                print()
                
            except ValueError:
                print(entry)
    
    def show_api_calls(self):
        """Show API call statistics"""
        api_calls = self.filter_entries(search="API call:")
        
        print("=" * 80)
        print(f"API CALLS ({len(api_calls)} total)")
        print("=" * 80)
        
        if not api_calls:
            print("No API calls found.\n")
            return
        
        # Parse and group by endpoint
        endpoints: Dict[str, List[float]] = defaultdict(list)
        
        for entry in api_calls:
            # Extract endpoint and duration
            # Format: "API call: METHOD url → status (X.XXs)"
            match = re.search(r'(GET|POST|PATCH|DELETE) ([^ ]+) .* \(([0-9.]+)s\)', entry.message)
            if match:
                method = match.group(1)
                url = match.group(2)
                duration = float(match.group(3))
                
                key = f"{method} {url}"
                endpoints[key].append(duration)
        
        # Show statistics
        for endpoint, durations in sorted(endpoints.items()):
            avg = sum(durations) / len(durations)
            min_d = min(durations)
            max_d = max(durations)
            print(f"\n{endpoint}")
            print(f"  Calls: {len(durations)}")
            print(f"  Avg:   {avg:.2f}s")
            print(f"  Min:   {min_d:.2f}s")
            print(f"  Max:   {max_d:.2f}s")
        print()


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze testing system logs")
    parser.add_argument('--dir', default='logs', help='Log directory (default: logs)')
    parser.add_argument('--service', help='Filter by service name')
    parser.add_argument('--date', help='Filter by date (YYYYMMDD)')
    parser.add_argument('--level', help='Filter by level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--search', help='Search for text in logs')
    parser.add_argument('--errors', action='store_true', help='Show errors only')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--timeline', action='store_true', help='Show activity timeline')
    parser.add_argument('--api', action='store_true', help='Show API call statistics')
    parser.add_argument('--limit', type=int, default=50, help='Limit results (default: 50)')
    parser.add_argument('--context', type=int, default=2, help='Context lines for search (default: 2)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LogAnalyzer(args.dir)
    analyzer.load_logs(service=args.service, date=args.date)
    
    if not analyzer.entries:
        print("❌ No log entries loaded. Exiting.")
        return
    
    # Show requested views
    if args.stats:
        analyzer.show_stats()
    
    if args.timeline:
        analyzer.show_timeline()
    
    if args.api:
        analyzer.show_api_calls()
    
    if args.errors:
        analyzer.show_errors(limit=args.limit)
    
    if args.search:
        analyzer.search(args.search, context=args.context)
    
    # Default: show stats and recent errors
    if not any([args.stats, args.timeline, args.api, args.errors, args.search]):
        analyzer.show_stats()
        analyzer.show_errors(limit=20)


if __name__ == "__main__":
    main()