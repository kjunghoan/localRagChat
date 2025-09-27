"""
Main application configuration that combines all config modules.
"""

from dataclasses import dataclass
from pathlib import Path

from .models import SupportedModel
from .chat import ChatConfig
from .debug import DebugConfig


@dataclass
class ModelConfig:
    """Model-related configuration"""

    # Model selection using enum
    model: SupportedModel = SupportedModel.MISTRAL_7B_INSTRUCT_V03
    # model: SupportedModel = SupportedModel.DIALOGPT_MEDIUM

    # Model loading settings
    use_quantization: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"


@dataclass
class AppConfig:
    """Main application configuration"""

    # Component configurations
    model: ModelConfig
    chat: ChatConfig
    debug: DebugConfig

    # Data paths
    data_dir: Path = Path("data")

    # Storage configuration
    storage_type: str = "pgvector"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

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
        config.chat.context_messages = 4
        config.chat.max_tokens = 300
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
        (self.data_dir / "vector_store").mkdir(exist_ok=True)
