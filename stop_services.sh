#!/bin/bash

echo "🛑 Stopping all services..."

pkill -f "test_bot_headless.py"
pkill -f "prepare_test_runs.py"
pkill -f "run_test_executions.py"
pkill -f "run_test_evaluations.py"

sleep 2

# Check if any are still running
RUNNING=$(ps aux | grep python | grep -E "test_bot|prepare|execute|evaluate" | grep -v grep)

if [ -z "$RUNNING" ]; then
    echo "✅ All services stopped"
else
    echo "⚠️  Some processes still running:"
    echo "$RUNNING"
fi
