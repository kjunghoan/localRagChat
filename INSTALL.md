# Installation Guide

Complete installation instructions for Local RAG Chat.

## Prerequisites

### System Requirements
- **Python 3.11 or 3.12** (3.10 might work, 3.13+ not tested)
- **8 GB+ RAM** recommended (4 GB minimum)
- **10 GB+ free disk space** (for models and data)
- **Internet connection** (for initial model downloads)

### Platform Support
- ✅ Linux (tested)
- ⚠️ macOS (tested, will be slow)
- ⚠️ Windows (should work, not tested)

## Quick Install (Experienced Users)

```bash
git clone <repository-url>
cd localRagChat
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
cp .env.example .env  # Edit with your HuggingFace token
python3 main.py
```

## Detailed Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd localRagChat
```

### Step 2: Python Virtual Environment
**Create virtual environment:**
```bash
python3 -m venv venv
```

**Activate virtual environment:**
```bash
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Verify activation:**
Your terminal prompt should now show `(venv)` at the beginning.

### Step 3: Install Dependencies
**Install pip-tools:**
```bash
pip install pip-tools
```

**Compile and install requirements:**
```bash
pip-compile requirements.in
pip install -r requirements.txt
```

**⚠️ This will take 5–10 minutes** - it's downloading PyTorch and other ML libraries.

### Step 4: HuggingFace Setup
**Get a HuggingFace token:**
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Go to Settings → Access Tokens
4. Create a new token (read access is fine)

**Set up environment file:**
```bash
cp .env.example .env
```

Edit `.env` file:
```
HF_TOKEN=<your_token_here>
```

### Step 5: First Run
```bash
python3 main.py
```

**First run will:**
- Download ~2 GB Mistral model (takes 5–15 minutes)
- Create data directories
- Initialize ChromaDB database

## Configuration

### Switching Models
Edit `src/configs/app.py`:
```python
# For DialoGPT (smaller, faster):
model: SupportedModel = SupportedModel.DIALOGPT_MEDIUM

# For Mistral (larger, better quality):
model: SupportedModel = SupportedModel.MISTRAL_7B_INSTRUCT_V03
```

### Adjusting Memory Usage
Edit chat config for your system:
```python
# In src/configs/chat.py
max_tokens: int = 150        # Smaller = faster but shorter responses
active_limit: int = 50       # Smaller = less memory usage(vram)
context_messages: int = 4    # Smaller = faster processing(cpu)
```

### Disabling Quantization
If you have lots of VRAM:
```python
# In src/configs/app.py
use_quantization: bool = False  # Better quality, more memory
```

## Troubleshooting

### Common Issues

**"pip-compile: command not found"**
```bash
pip install pip-tools
```

**"No module named 'torch'"**
```bash
# Requirements didn't install properly
pip install -r requirements.txt
```

**"ModuleNotFoundError: No module named 'src'"**
```bash
# Make sure you're in the project root directory
pwd  # Should show /path/to/localRagChat
python3 main.py  # Not python src/main.py
```

**"HuggingFace token required"**
- Create `.env` file with your token
- Get token from huggingface.co/settings/tokens

**"CUDA out of memory" or GPU errors**
```python
# Edit src/configs/app.py
use_quantization: bool = True
device_map: str = "cpu"  # Force CPU usage
```

**Model downloads failing**
- Check internet connection
- HuggingFace might be down, try again later
- Some models require acceptance of license terms

**Slow responses**
- Use DialoGPT instead of Mistral
- Reduce max_tokens to 100-150
- Enable quantization
- Reduce context_messages to 4

### Platform-Specific Issues

**macOS: "xcrun: error"**
```bash
xcode-select --install
```

**Linux: Missing build tools**
```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install build-essential python3-dev

# CentOS/RHEL:
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

**Windows: Long path issues**
- Run as Administrator
- Enable long path support in Windows

## Verification

### Test Installation
```bash
python3 main.py
```

**You should see:**
```
🤖 Local RAG Chat starting:
   Model: [Model Name]
   Quantization: Enabled/Disabled
✅ Setup Complete!
Chat started! Type 'exit' 'q' or <C-c> to end.
```

### Test Conversation
```
User: hello
Chatbot: [Response from AI]
```

### Test Memory
Have a conversation with messages, then exit. You should see:
```
💾 Session saved to vector store: [ID]...
```

## Getting Help

### Debug Mode
Enable debug logging:
```python
# In main.py, change:
config = AppConfig.debug_mode()
```

### Check Logs
Look for error messages in the terminal output.

### File Issues
Create a GitHub issue with:
- Your operating system
- Python version (`python3 --version`)
- Error message (full trace back)
- What you were trying to do

## What's Next?

Once installed successfully:
1. Read README.md for usage instructions
2. Try different models in the config
3. Experiment with memory limits
4. Check out the conversation viewer: `python3 view_conversations.py`
