import os
from dataclasses import dataclass
from enum import Enum


class SupportedModel(Enum):
    MISTRAL_7B_INSTRUCT_V03 = "mistral-7b-instruct-v0.3"
    DIALOGPT_LARGE = "dialogpt-large"
    DIALOGPT_MEDIUM = "dialogpt-medium"

    @property
    def hf_name(self) -> str:
        mapping = {
            SupportedModel.MISTRAL_7B_INSTRUCT_V03: "mistralai/Mistral-7B-Instruct-v0.3",
            SupportedModel.DIALOGPT_LARGE: "microsoft/DialoGPT-large",
            SupportedModel.DIALOGPT_MEDIUM: "microsoft/DialoGPT-medium",
        }
        return mapping[self]

    @property
    def model_type(self) -> str:
        mapping = {
            SupportedModel.MISTRAL_7B_INSTRUCT_V03: "mistral",
            SupportedModel.DIALOGPT_LARGE: "dialogpt",
            SupportedModel.DIALOGPT_MEDIUM: "dialogpt",
        }
        return mapping[self]

    @property
    def display_name(self) -> str:
        mapping = {
            SupportedModel.MISTRAL_7B_INSTRUCT_V03: "Mistral 7B Instruct v0.3",
            SupportedModel.DIALOGPT_LARGE: "DialoGPT Large",
            SupportedModel.DIALOGPT_MEDIUM: "DialoGPT Medium",
        }
        return mapping[self]


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Config:
    model: SupportedModel
    use_quantization: bool
    torch_dtype: str
    device_map: str
    max_tokens: int
    max_length: int
    temperature: float
    do_sample: bool
    context_messages: int
    active_limit: int
    log_level: LogLevel
    database_url: str
    embedding_model: str

    def __post_init__(self):
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")

    @classmethod
    def from_env(cls) -> "Config":
        model_name = os.getenv("AI_MODEL")
        if not model_name:
            raise ValueError("AI_MODEL environment variable is required")

        try:
            model = SupportedModel(model_name)
        except ValueError:
            raise ValueError(
                f"Invalid AI_MODEL: {model_name}. "
                f"Options: {[m.value for m in SupportedModel]}"
            )

        quant_str = os.getenv("USE_QUANTIZATION", "true").lower()
        use_quantization = quant_str in ("true", "1", "yes")
        torch_dtype = os.getenv("TORCH_DTYPE", "float16")
        device_map = os.getenv("DEVICE_MAP", "auto")

        max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "300"))
        max_length = int(os.getenv("CHAT_MAX_LENGTH", "1024"))
        temperature = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
        do_sample_str = os.getenv("CHAT_DO_SAMPLE", "true").lower()
        do_sample = do_sample_str in ("true", "1", "yes")
        context_messages = int(os.getenv("CHAT_CONTEXT_MESSAGES", "6"))
        active_limit = int(os.getenv("CHAT_ACTIVE_LIMIT", "100"))

        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        try:
            log_level = LogLevel(log_level_str)
        except ValueError:
            raise ValueError(
                f"Invalid LOG_LEVEL: {log_level_str}. "
                f"Options: {[l.value for l in LogLevel]}"
            )

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        return cls(
            model=model,
            use_quantization=use_quantization,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_tokens=max_tokens,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            context_messages=context_messages,
            active_limit=active_limit,
            log_level=log_level,
            database_url=database_url,
            embedding_model=embedding_model,
        )

    def get_model_info(self) -> dict:
        return {
            "name": self.model.display_name,
            "hf_name": self.model.hf_name,
            "type": self.model.model_type,
            "quantization": self.use_quantization,
            "max_tokens": self.max_tokens,
        }
