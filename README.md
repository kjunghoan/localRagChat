# Local RAG Chat

A local-first conversational AI system with persistent memory and modular architecture.
No external APIs, no internet dependency after initial setup.


## Overview

Local RAG Chat is part of a microservice-based AI companion that runs entirely on your network.
It features rolling conversation memory, vector-based storage for long-term recall, and a clean
factory pattern architecture for easy extensibility.

## Key Features

- **🤖 Multiple AI Models**: Support for Multiple interchangeable AI models
- **🧠 Smart Memory Management**: Rolling context with RAM→Disk spillover
- **📚 Vector Storage**: ChromaDB integration for conversation history and retrieval
- **⚙️ Configurable**: Clean config system with model/chat/debug settings
- **🔄 Local-First**: No external API calls, with caveats, everything runs on your hardware
- **💾 Persistent Sessions**: Conversations saved and recoverable across restarts

## Quick Start

```bash
# See INSTALL.md for detailed setup instructions
git clone git@github.com:kjunghoan/localRagChat.git
cd localRagChat
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your HuggingFace token
python3 main.py
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
Edit `src/configs/app.py`:
```python
# Switch between supported models
model: SupportedModel = SupportedModel.MISTRAL_7B_INSTRUCT_V03
# or
model: SupportedModel = SupportedModel.DIALOGPT_MEDIUM
```

### Memory Management
The system automatically manages conversation memory:
- **Active Memory**: Recent messages (configurable limit)
- **Session Storage**: Older messages from current session
- **Vector Storage**: Persistent storage across sessions

### Viewing Conversations
```bash
# Browse saved conversations
python3 view_conversations.py

# Debug storage internals
python3 debug_storage.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Core App      │────│  Model Factory   │────│ AI Models       │
│                 │    │                  │    │ (Mistral/GPT)   │
└─────────────────┘    └──────────────────┘    │ Local model(tbd)│
         │                                     └─────────────────┘
         ├── ┌─────────────────┐    ┌──────────────────┐
         │   │ Rolling Memory  │────│ Vector Storage   │
         │   │                 │    │   (ChromaDB)     │
         │   └─────────────────┘    └──────────────────┘
         │
         └── ┌─────────────────┐
             │ Config System   │
             │ (Modular)       │
             └─────────────────┘
```

### Key Components

- **Core App** (`src/core/`): Application orchestration and chat loop
- **Models** (`src/models/`): AI model implementations with unified interface
- **Memory** (`src/memory/`): Rolling conversation memory with spillover management
- **Storage** (`src/storage/`): Vector storage interface and ChromaDB implementation
- **Factories** (`src/factories/`): Component creation with dependency injection
- **Configs** (`src/configs/`): Modular configuration system

## Configuration

### Supported Models
- **Mistral 7B Instruct v0.3**: High-quality responses, instruction-tuned
- **DialoGPT Large**: Faster, conversational style
- **DialoGPT Medium**: Lightest option, good for limited resources

### Memory Settings
```python
# In src/configs/chat.py
max_tokens: int = 300        # Response length
active_limit: int = 100      # Messages in active memory  
context_messages: int = 6    # AI context window
```

### Performance Tuning
```python
# In src/configs/app.py
use_quantization: bool = True   # Reduce memory usage
torch_dtype: str = "float16"    # Optimize for your hardware
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
│   ├── storage/       # Vector storage
│   ├── factories/     # Component factories
│   ├── configs/       # Configuration system
│   └── utils/         # Utilities (logging, etc.)
├── data/              # Generated data directory
├── INSTALL.md         # Installation guide
└── main.py           # Entry point
```

### Adding New Models
1. Implement `TransformerModelInterface` in `src/models/`
2. Add to `SupportedModel` enum in `src/configs/models.py`
3. Register in `ModelFactory` registry
4. Update model mappings

### Extending Storage
1. Implement `VectorStoreInterface` in `src/storage/`
2. Register in `StorageFactory`
3. Add configuration options

## Roadmap

- **v1**: Proof of concept context expansion using disk
- **v2**: Cache invalidation for oldest unreferenced data
- **v3**: Memory hierarchies and integration with other services
- **v4**: MCP with agent read/writes to other services
