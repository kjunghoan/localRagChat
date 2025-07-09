from typing import Dict, Type
from ..interfaces.model import TransformerModelInterface, ModelConfig
from ..models import MistralModel, DialoGPTModel


class ModelFactory:
    """
    Factory for creating model instances based on configuration.
    """

    # Registry of available model types
    _model_registry: Dict[str, Type[TransformerModelInterface]] = {
        "mistral": MistralModel,
        "dialogpt": DialoGPTModel,
    }

    @classmethod
    def create(cls, model_type: str, config: ModelConfig) -> TransformerModelInterface:
        """
        Create a model instance based on type and configuration.

        Args:
            model_type: Type of model ("mistral", "dialogpt", etc.)
            config: ModelConfig with model-specific settings

        Returns:
            Initialized model instance (not yet loaded)

        Raises:
            ValueError: If model_type is not supported
        """
        if config is None:
            raise ValueError("Model configuration cannot be None")
            
        if model_type not in cls._model_registry:
            available = ", ".join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available: {available}"
            )

        model_class = cls._model_registry[model_type]
        return model_class(config)

    @classmethod
    def create_and_load(
        cls, model_type: str, config: ModelConfig
    ) -> TransformerModelInterface:
        """
        Create and immediately load a model.

        Args:
            model_type: Type of model to create
            config: Model configuration

        Returns:
            Loaded model ready for inference
        """
        model = cls.create(model_type, config)
        model.load()
        return model

    @classmethod
    def available_models(cls) -> list[str]:
        """Get list of available model types"""
        return list(cls._model_registry.keys())

    @classmethod
    def register_model(
        cls, model_type: str, model_class: Type[TransformerModelInterface]
    ) -> None:
        """
        Register a new model type (for extensibility).

        Args:
            model_type: String identifier for the model
            model_class: Class that implements ModelInterface
        """
        cls._model_registry[model_type] = model_class


# Convenience functions for common model creation patterns
def create_mistral(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", **kwargs
) -> TransformerModelInterface:
    """Create a Mistral model with sensible defaults"""
    config = ModelConfig(name=model_name, **kwargs)
    return ModelFactory.create_and_load("mistral", config)


def create_dialogpt(
    model_name: str = "microsoft/DialoGPT-large", **kwargs
) -> TransformerModelInterface:
    """Create a DialoGPT model with sensible defaults"""
    config = ModelConfig(
        name=model_name,
        use_quantization=False,  # DialoGPT is smaller, usually doesn't need quantization
        **kwargs,
    )
    return ModelFactory.create_and_load("dialogpt", config)
