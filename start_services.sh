#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Full path to Python in virtual environment
PYTHON="$DIR/env/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON" ]; then
    echo "❌ Virtual environment Python not found at: $PYTHON"
    echo "   Please create virtual environment first: python3 -m venv env"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Stop any existing processes
echo "🛑 Stopping existing processes..."
pkill -f "test_bot_headless.py"
pkill -f "prepare_test_runs.py"
pkill -f "run_test_executions.py"
pkill -f "run_test_evaluations.py"

# Wait a moment
sleep 2

# Start all services with full path
echo "🚀 Starting services..."

echo "   ▶️  Starting test_bot_headless.py..."
nohup "$PYTHON" test_bot_headless.py > logs/testbot.log 2>&1 & disown

echo "   ▶️  Starting prepare_test_runs.py..."
nohup "$PYTHON" prepare_test_runs.py > logs/prepare.log 2>&1 & disown

sleep 3
echo "   ▶️  Starting run_test_executions.py..."
nohup "$PYTHON" run_test_executions.py > logs/execute.log 2>&1 & disown

echo "   ▶️  Starting run_test_evaluations.py..."
nohup "$PYTHON" run_test_evaluations.py > logs/evaluate.log 2>&1 & disown

# Wait for processes to start
sleep 3

# Show running processes
echo ""
echo "✅ Services started! Running processes:"
ps aux | grep python | grep -E "test_bot|prepare|execute|evaluate" | grep -v grep

echo ""
echo "📊 Monitor logs with:"
echo "   tail -f logs/testbot.log"
echo "   tail -f logs/prepare.log"
echo "   tail -f logs/execute.log"
echo "   tail -f logs/evaluate.log"