from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

ConversationHistory = List[Dict[str, str]]


@dataclass
class ModelConfig:
    """Configuration for model loading and inference"""

    name: str
    use_quantization: bool = True
    torch_dtype: str = "float16"
    device_map: str = "auto"

    # Model-specific parameters that subclasses might need
    max_tokens: int = 150
    temperature: float = 0.7
    do_sample: bool = True


class TransformerModelInterface(ABC):
    """
    Abstract interface for transformer-based chat models.

    This interface assumes:
    - Models use the HuggingFace transformers library
    - Models have tokenizers for text processing
    - Models follow the transformer architecture pattern

    For non-transformer models, a different interface would be needed.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def format_prompt(self, conversation_history: ConversationHistory) -> str:
        """
        Format conversation history into model-specific prompt format.

        Args:
            conversation_history: List of {"role": str, "content": str} messages

        Returns:
            Formatted prompt string ready for the model
        """
        pass

    @abstractmethod
    def parse_response(self, raw_output: str, debug: bool = False) -> str:
        """
        Extract clean response from model's raw output.

        Args:
            raw_output: Full text output from model generation
            debug: Whether to print debug information

        Returns:
            Clean response text for the user
        """
        pass

    @abstractmethod
    def generate_raw(self, prompt: str, debug: bool = False) -> str:
        """
        Generate raw response from formatted prompt.

        Args:
            prompt: Formatted prompt string
            debug: Whether to print debug information

        Returns:
            Raw model output (before parsing)
        """
        pass

    def generate_response(
        self, conversation_history: ConversationHistory, debug: bool = False
    ) -> str:
        """
        Complete generation pipeline: format → generate → parse.
        This is the main public method that orchestrators will call.

        Args:
            conversation_history: List of conversation messages
            debug: Whether to enable debug output

        Returns:
            Clean response ready for user
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format the prompt for this specific model
        prompt = self.format_prompt(conversation_history)

        if debug:
            print(f"🔍 DEBUG - Formatted prompt length: {len(prompt)}")

        # Generate raw response
        raw_output = self.generate_raw(prompt, debug)

        # Parse and clean the response
        clean_response = self.parse_response(raw_output, debug)

        return clean_response

    @property
    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded"""
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get basic information about the loaded model"""
        return {
            "name": self.config.name,
            "quantization": self.config.use_quantization,
            "loaded": self.is_loaded,
            "device": str(next(self.model.parameters()).device)
            if self.is_loaded
            else "not loaded",
        }
