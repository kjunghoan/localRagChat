# Local RAG Chat

A local-first conversational AI system with persistent memory and simplified architecture.
No external APIs, no internet dependency after initial setup.


## Overview

Local RAG Chat is part of a microservice-based AI companion that runs entirely on your network.
It features rolling conversation memory, vector-based storage for long-term recall, and a clean
environment-based configuration system for easy customization.

## Key Features

- **🤖 Multiple AI Models**: Support for multiple interchangeable AI models
- **🧠 Smart Memory Management**: Rolling context with RAM→Database spillover
- **📚 Vector Storage**: PostgreSQL + pgvector for conversation history and retrieval
- **⚙️ Configurable**: Environment-based configuration via .env file
- **🔄 Local-First**: No external API calls, with caveats, everything runs on your hardware
- **💾 Persistent Sessions**: Conversations saved and recoverable across restarts

## Quick Start

```bash
# See INSTALL.md for detailed setup instructions
git clone git@github.com:kjunghoan/localRagChat.git
cd localRagChat
uv sync  # Install dependencies with uv
cp .env.example .env  # Configure settings (HuggingFace token, model choice, etc.)
uv run main.py
```

## Usage

### Basic Conversation
```
User: Hello! How are you today?
Chatbot: Hello! I'm doing well, thank you for asking. How can I help you today?

User: What's the weather like?
Chatbot: I don't have access to real-time weather data, but I'd be happy to help you with other questions!
```

### Model Switching
Edit `.env`:
```bash
# Switch between supported models
AI_MODEL=mistralai/Mistral-7B-Instruct-v0.3
# or
AI_MODEL=microsoft/DialoGPT-medium
```

### Memory Management
The system automatically manages conversation memory:
- **Active Memory**: Recent messages (configurable limit)
- **Session Storage**: Older messages from current session
- **Vector Storage**: Persistent storage across sessions

### Exiting and Session Management
```bash
# Type 'exit', 'quit', or press Ctrl+C to safely exit
# Conversations are automatically saved to PostgreSQL
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Core App      │────│ AI Models       │
│                 │    │ (Mistral/GPT)   │
└─────────────────┘    │ Local model(tbd)│
         │             └─────────────────┘
         ├── ┌─────────────────┐    ┌──────────────────┐
         │   │ Rolling Memory  │────│ Vector Storage   │
         │   │                 │    │ (PostgreSQL +    │
         │   └─────────────────┘    │  pgvector)       │
         │                          └──────────────────┘
         └── ┌─────────────────┐
             │ Configuration   │
             │ (.env file)     │
             └─────────────────┘
```

### Key Components

- **Core App** (`src/core/`): Application orchestration and chat loop
- **Models** (`src/models/`): AI model implementations (Mistral, DialoGPT)
- **Memory** (`src/memory/`): Rolling conversation memory with spillover management
- **Storage** (`src/storage/`): PostgreSQL + pgvector implementation for conversation persistence
- **Config** (`src/config.py`): Unified configuration loaded from environment variables

## Configuration

### Supported Models
- **Mistral 7B Instruct v0.3**: High-quality responses, instruction-tuned
- **DialoGPT Large**: Faster, conversational style
- **DialoGPT Medium**: Lightest option, good for limited resources

### Memory Settings
```bash
# In .env
CHAT_MAX_TOKENS=300           # Response length
CHAT_ACTIVE_LIMIT=100         # Messages in active memory
CHAT_CONTEXT_MESSAGES=6       # AI context window
```

### Performance Tuning
```bash
# In .env
USE_QUANTIZATION=true         # Reduce memory usage
TORCH_DTYPE=float16           # Optimize for your hardware
LOG_LEVEL=INFO                # Logging verbosity (DEBUG, INFO, WARNING, ERROR)
```

## Requirements

- **Python 3.11 | 3.12**
- **8 GB+ RAM** (4 GB minimum with quantization)
- **10 GB+ disk space** (for models and conversations)
- **HuggingFace account** (free, for model downloads)

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Development

### Project Structure
```
localRagChat/
├── src/
│   ├── core/          # Application orchestration
│   ├── models/        # AI model implementations
│   ├── memory/        # Memory management
│   ├── storage/       # Vector storage (PostgreSQL + pgvector)
│   ├── utils/         # Utilities (logging, etc.)
│   ├── config.py      # Unified configuration
│   └── auth.py        # Authentication (HuggingFace token)
├── tests/             # Test suite
├── .env.example       # Example environment configuration
├── INSTALL.md         # Installation guide
└── main.py            # Entry point
```

### Adding New Models
1. Implement `ModelInterface` in `src/models/`
2. Update `_create_model()` in `src/core/app.py` to support the new model
3. Set `AI_MODEL` in `.env` to the HuggingFace model name

### Extending Storage
1. Create new storage implementation in `src/storage/`
2. Update `_create_storage()` in `src/core/app.py`
3. Add required configuration to `VectorStoreConfig`

## Roadmap

- **v1**: Proof of concept context expansion using disk
- **v2**: Cache invalidation for oldest unreferenced data
- **v3**: Memory hierarchies and integration with other services
- **v4**: MCP with agent read/writes to other services
