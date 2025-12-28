#!/bin/bash
# A2A Agent Startup Script
# Uses PYTHONPATH to ensure modules can be found

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"

cd "$SCRIPT_DIR"

# Check Python
echo "Python path: $(which python3)"
python3 --version

# Check if package is installed
echo "Checking package installation..."
python3 -c "import a2a_adapter; print('✓ a2a_adapter installed')" 2>/dev/null || {
    echo "❌ a2a_adapter not installed, please run:"
    echo "  source .venv/bin/activate && pip install -e ."
    exit 1
}

# Run specified example
if [ $# -eq 0 ]; then
    echo "Usage: $0 <agent_type>"
    echo "Supported agent types: n8n, crewai, langchain"
    exit 1
fi

AGENT_TYPE="$1"
SCRIPT_FILE=""

case "$AGENT_TYPE" in
    n8n)
        SCRIPT_FILE="examples/01_single_n8n_agent.py"
        ;;
    crewai)
        SCRIPT_FILE="examples/02_single_crewai_agent.py"
        ;;
    langchain)
        SCRIPT_FILE="examples/03_single_langchain_agent.py"
        ;;
    *)
        echo "Unknown agent type: $AGENT_TYPE"
        echo "Supported types: n8n, crewai, langchain"
        exit 1
        ;;
esac

echo "Starting $AGENT_TYPE agent..."
echo "Script file: $SCRIPT_FILE"
echo "PYTHONPATH: $PYTHONPATH"
echo "Press Ctrl+C to stop server"
echo "----------------------------------------"

# Run script
python3 "$SCRIPT_FILE" "$@"
