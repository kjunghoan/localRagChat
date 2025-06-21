from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model-related configuration"""

    name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    use_quantization: bool = True
    torch_dtype: str = "float16"
    device_map: str = "auto"


@dataclass
class ChatConfig:
    """Chat behavior configuration"""

    max_tokens: int = 150
    max_length: int = 1024
    context_messages: int = 6
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class DebugConfig:
    """Debug and logging configuration"""

    enabled: bool = False
    verbose_loading: bool = False
    show_token_counts: bool = False
    show_full_responses: bool = False


@dataclass
class AppConfig:
    """Main application configuration"""

    model: ModelConfig
    chat: ChatConfig
    debug: DebugConfig

    # Data paths
    data_dir: Path = Path("data")
    conversations_dir: Path = Path("data/conversations")
    embeddings_dir: Path = Path("data/embeddings")

    @classmethod
    def default(cls) -> "AppConfig":
        """Create default configuration"""
        return cls(model=ModelConfig(), chat=ChatConfig(), debug=DebugConfig())

    @classmethod
    def debug_mode(cls) -> "AppConfig":
        """Create configuration with debug enabled"""
        config = cls.default()
        config.debug.enabled = True
        config.debug.verbose_loading = True
        config.debug.show_token_counts = True
        return config

    @classmethod
    def development(cls) -> "AppConfig":
        """Configuration optimized for development"""
        config = cls.default()
        config.debug.enabled = True
        config.chat.context_messages = 4  # Smaller context for faster testing
        config.chat.max_tokens = 100
        return config

    @classmethod
    def production(cls) -> "AppConfig":
        """Configuration optimized for production use"""
        config = cls.default()
        config.chat.max_tokens = 200
        config.chat.context_messages = 8
        return config

    def ensure_directories(self) -> None:
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
