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
uv sync  # Install all dependencies with uv
cp .env.example .env  # Edit with your HuggingFace token and settings
uv run main.py
```

## Detailed Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd localRagChat
```

### Step 2: Install uv (Package Manager)

**Install uv (if not already installed):**

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, see [uv documentation](https://docs.astral.sh/uv/).

### Step 3: Install Dependencies

**Install all dependencies:**

```bash
uv sync
```

**⚠️ This will take 5–10 minutes** - it's downloading PyTorch and other ML libraries.
`uv` will automatically create a virtual environment and install all dependencies.

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

```bash
# Required
HF_TOKEN=<your_token_here>
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Model selection (choose one)
AI_MODEL=mistralai/Mistral-7B-Instruct-v0.3
# AI_MODEL=microsoft/DialoGPT-medium

# Optional configuration (see .env.example for all options)
CHAT_MAX_TOKENS=300
LOG_LEVEL=INFO
```

### Step 5: PostgreSQL Setup

**Install PostgreSQL with pgvector extension:**

```bash
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-15-pgvector  # Adjust version number as needed

# macOS (with Homebrew):
brew install postgresql@15
brew install pgvector
```

**Create database:**

```bash
# Start PostgreSQL service
sudo systemctl start postgresql  # Linux
brew services start postgresql@15  # macOS

# Create database and enable pgvector
sudo -u postgres psql
CREATE DATABASE localragchat;
\c localragchat
CREATE EXTENSION vector;
\q
```

Update `DATABASE_URL` in your `.env` file accordingly.

### Step 6: First Run

```bash
uv run main.py
```

**First run will:**

- Download ~2 GB Mistral model (takes 5–15 minutes)
- Create PostgreSQL table with pgvector extension
- Initialize session

## Configuration

### Switching Models

Edit `.env`:

```bash
# For DialoGPT (smaller, faster):
AI_MODEL=microsoft/DialoGPT-medium

# For Mistral (larger, better quality):
AI_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

### Adjusting Memory Usage

Edit `.env` for your system:

```bash
CHAT_MAX_TOKENS=150          # Smaller = faster but shorter responses
CHAT_ACTIVE_LIMIT=50         # Smaller = less memory usage (RAM)
CHAT_CONTEXT_MESSAGES=4      # Smaller = faster processing (CPU)
```

### Disabling Quantization

If you have lots of VRAM:

```bash
# In .env
USE_QUANTIZATION=false       # Better quality, more memory
```

## Troubleshooting

### Common Issues

**"uv: command not found"**

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
```

**"No module named 'torch'"**

```bash
# Dependencies didn't install properly
uv sync
```

**"ModuleNotFoundError: No module named 'src'"**

```bash
# Make sure you're in the project root directory
pwd  # Should show /path/to/localRagChat
uv run main.py  # Not python src/main.py
```

**"HuggingFace token required"**

- Create `.env` file with your token
- Get token from huggingface.co/settings/tokens

**"CUDA out of memory" or GPU errors**

```bash
# Edit .env
USE_QUANTIZATION=true
DEVICE_MAP=cpu  # Force CPU usage
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
uv run main.py
```

**You should see:**

```txt
🤖 Local RAG Chat starting:
   Model: [Model Name]
   Quantization: Enabled/Disabled
✅ Setup Complete!
Chat started! Type 'exit' 'q' or <C-c> to end.
```

### Test Conversation

```txt
User: hello
Chatbot: [Response from AI]
```

### Test Memory

Have a conversation with messages, then exit. You should see:

```txt
💾 Session saved to vector store: [ID]...
```

## Getting Help

### Debug Mode

Enable debug logging:

```bash
# In .env, change:
LOG_LEVEL=DEBUG
```

### Check Logs

Look for error messages in the terminal output.

### File Issues

Create a GitHub issue with:

- Your operating system
- Python version (`python3 --version`)
- Error message (full traceback)
- What you were trying to do

## What's Next?

Once installed successfully:

1. Read README.md for usage instructions
2. Try different models by editing `.env`
3. Experiment with memory limits in `.env`
4. Conversations are automatically saved to PostgreSQL
