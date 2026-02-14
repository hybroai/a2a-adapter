# Getting Started with Local Debugging 🚀

> Step-by-step guide to debug A2A Adapters SDK locally

## 🎯 Goal

Complete environment setup and run your first debug test in 5 minutes.

## 📋 Prerequisites

- Python 3.9 or higher
- Terminal/command line access
- (Optional) VS Code or PyCharm

## 🚀 Step 1: Environment Setup

### Method 1: Automatic Setup (Recommended)

```bash
cd "/a2a-adapter"

# Run setup script
./setup_dev.sh
```

This script will automatically:

- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install development dependencies
- ✅ Run validation tests

### Method 2: Manual Setup

```bash
cd "/a2a-adapter"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import a2a_adapter; print('✅ Installation successful!')"
```

## 🧪 Step 2: Run Your First Test

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run the simplest test
python debug_scripts/01_simple_test.py
```

**Expected output:**

```
============================================================
🧪 A2A Adapters - Simple Local Test
============================================================

📦 Step 1: Loading callable adapter...
✅ Adapter loaded successfully

📝 Step 2: Creating test message...
✅ Test message created successfully

🚀 Step 3: Calling adapter.handle()...
✅ Agent received message: hello world
✅ Call successful

📊 Step 4: Result analysis
   - Role: assistant
   - Content type: <class 'list'>
   - Content: Echo: HELLO WORLD

============================================================
✅ Test completed! All functions working normally
============================================================
```

If you see this output, congratulations! Environment setup successful 🎉

## 🌐 Step 3: Test Full Server

### Terminal 1: Start Server

```bash
# Start any agent server
./run_agent.sh n8n        # N8n agent
./run_agent.sh crewai     # CrewAI agent
./run_agent.sh langchain  # LangChain agent
```

### Terminal 2: Test Server

```bash
# Test with client
python examples/single_agent_client.py
```

## 🔍 Step 4: Debug with VS Code

1. Open project in VS Code
2. Set breakpoint in `a2a_adapter/adapter.py`
3. Press F5 to start debugging
4. Select "🧪 Debug: Simple Test"

## 📝 Troubleshooting

- **ModuleNotFoundError**: Run `source venv/bin/activate && pip install -e .`
- **Port already in use**: `lsof -i :9000` to find process, then `kill -9 <PID>`
- **Permission denied**: Make scripts executable with `chmod +x run_agent.sh`

## 🎉 Congratulations!

You now have a fully functional A2A Adapters development environment! 🚀
