"""
Model configuration definitions and supported models.
"""

from enum import Enum
from dataclasses import dataclass


class SupportedModel(Enum):
    """Enumeration of supported AI models"""

    MISTRAL_7B_INSTRUCT_V03 = "mistral-7b-instruct-v0.3"
    DIALOGPT_LARGE = "dialogpt-large"
    DIALOGPT_MEDIUM = "dialogpt-medium"

    @property
    def hf_name(self) -> str:
        """Get the HuggingFace model name for loading"""
        mapping = {
            SupportedModel.MISTRAL_7B_INSTRUCT_V03: "mistralai/Mistral-7B-Instruct-v0.3",
            SupportedModel.DIALOGPT_LARGE: "microsoft/DialoGPT-large",
            SupportedModel.DIALOGPT_MEDIUM: "microsoft/DialoGPT-medium",
        }
        return mapping[self]

    @property
    def model_type(self) -> str:
        """Get the factory registry key"""
        mapping = {
            SupportedModel.MISTRAL_7B_INSTRUCT_V03: "mistral",
            SupportedModel.DIALOGPT_LARGE: "dialogpt",
            SupportedModel.DIALOGPT_MEDIUM: "dialogpt",
        }
        return mapping[self]

    @property
    def display_name(self) -> str:
        """Get a human-readable display name"""
        mapping = {
            SupportedModel.MISTRAL_7B_INSTRUCT_V03: "Mistral 7B Instruct v0.3",
            SupportedModel.DIALOGPT_LARGE: "DialoGPT Large",
            SupportedModel.DIALOGPT_MEDIUM: "DialoGPT Medium",
        }
        return mapping[self]


@dataclass
class ModelConfig:
    """Configuration for AI model loading and behavior"""

    # Model selection
    model: SupportedModel = SupportedModel.MISTRAL_7B_INSTRUCT_V03

    # Quantization settings
    use_quantization: bool = True
    torch_dtype: str = "auto"

    # Device and memory management
    device_map: str = "auto"
    trust_remote_code: bool = False

    # Generation parameters (will be moved to ChatConfig later)
    max_tokens: int = 300
    temperature: float = 0.7
    do_sample: bool = True

    @property
    def hf_model_name(self) -> str:
        """Get the HuggingFace model name"""
        return self.model.hf_name

    @property
    def factory_type(self) -> str:
        """Get the model factory type"""
        return self.model.model_type

    def get_model_info(self) -> dict:
        """Get comprehensive model information"""
        return {
            "name": self.model.display_name,
            "hf_name": self.hf_model_name,
            "type": self.factory_type,
            "quantization": self.use_quantization,
            "max_tokens": self.max_tokens,
        }
