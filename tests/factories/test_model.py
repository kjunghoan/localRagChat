"""
Tests for the model factory implementation.
"""

import pytest
from unittest.mock import Mock, patch
from src.factories.model import ModelFactory, create_mistral, create_dialogpt
from src.interfaces.model import TransformerModelInterface, ModelConfig
from src.models.mistral import MistralModel
from src.models.dialogpt import DialoGPTModel


class TestModelFactory:
    """Test the ModelFactory class"""

    def test_create_valid_model_type(self):
        """Test creating models with valid model types"""
        config = ModelConfig(name="test-model")
        
        mistral = ModelFactory.create("mistral", config)
        dialogpt = ModelFactory.create("dialogpt", config)
        
        assert isinstance(mistral, MistralModel)
        assert isinstance(dialogpt, DialoGPTModel)
        assert mistral.config == config
        assert dialogpt.config == config

    def test_create_unknown_model_type(self):
        """Test creating unknown model type raises ValueError"""
        config = ModelConfig(name="test-model")
        
        with pytest.raises(ValueError) as exc_info:
            ModelFactory.create("unknown", config)
        
        assert "Unknown model type 'unknown'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    @patch.object(MistralModel, 'load')
    def test_create_and_load(self, mock_load):
        """Test create_and_load method calls load on model"""
        config = ModelConfig(name="test-model")
        
        model = ModelFactory.create_and_load("mistral", config)
        
        assert isinstance(model, MistralModel)
        mock_load.assert_called_once()

    def test_register_model(self):
        """Test registering a new model type"""
        mock_model_class = Mock(spec=TransformerModelInterface)
        mock_instance = Mock(spec=TransformerModelInterface)
        mock_model_class.return_value = mock_instance
        
        # Register the new model
        ModelFactory.register_model("test_model", mock_model_class)
        
        # Test that we can create it
        config = ModelConfig(name="test")
        result = ModelFactory.create("test_model", config)
        
        assert result == mock_instance
        mock_model_class.assert_called_once_with(config)
        
        # Clean up
        del ModelFactory._model_registry["test_model"]

    def test_register_model_overwrites_existing(self):
        """Test that registering overwrites existing model types"""
        original_class = ModelFactory._model_registry["mistral"]
        mock_model_class = Mock(spec=TransformerModelInterface)
        mock_instance = Mock(spec=TransformerModelInterface)
        mock_model_class.return_value = mock_instance
        
        ModelFactory.register_model("mistral", mock_model_class)
        
        config = ModelConfig(name="test")
        result = ModelFactory.create("mistral", config)
        
        assert result == mock_instance
        mock_model_class.assert_called_once_with(config)
        
        # Clean up
        ModelFactory._model_registry["mistral"] = original_class

    def test_create_with_different_configs(self):
        """Test creating models with different configuration options"""
        config1 = ModelConfig(
            name="model1",
            use_quantization=True,
            torch_dtype="float16"
        )
        config2 = ModelConfig(
            name="model2",
            use_quantization=False,
            torch_dtype="float32"
        )
        
        model1 = ModelFactory.create("mistral", config1)
        model2 = ModelFactory.create("mistral", config2)
        
        assert model1.config == config1
        assert model2.config == config2
        assert model1.config.use_quantization != model2.config.use_quantization


class TestConvenienceFunctions:
    """Test the convenience functions for model creation"""

    @patch.object(ModelFactory, 'create_and_load')
    def test_create_mistral_with_defaults(self, mock_create_and_load):
        """Test create_mistral function with default parameters"""
        mock_model = Mock(spec=TransformerModelInterface)
        mock_create_and_load.return_value = mock_model
        
        result = create_mistral()
        
        mock_create_and_load.assert_called_once()
        call_args = mock_create_and_load.call_args
        
        assert call_args[0][0] == "mistral"  # model_type
        config = call_args[0][1]  # config
        assert config.name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert result == mock_model

    @patch.object(ModelFactory, 'create_and_load')
    def test_create_mistral_with_custom_params(self, mock_create_and_load):
        """Test create_mistral function with custom parameters"""
        mock_model = Mock(spec=TransformerModelInterface)
        mock_create_and_load.return_value = mock_model
        
        result = create_mistral(
            model_name="custom/model",
            use_quantization=False,
            torch_dtype="float32"
        )
        
        mock_create_and_load.assert_called_once()
        call_args = mock_create_and_load.call_args
        
        assert call_args[0][0] == "mistral"
        config = call_args[0][1]
        assert config.name == "custom/model"
        assert config.use_quantization == False
        assert config.torch_dtype == "float32"

    @patch.object(ModelFactory, 'create_and_load')
    def test_create_dialogpt_with_defaults(self, mock_create_and_load):
        """Test create_dialogpt function with default parameters"""
        mock_model = Mock(spec=TransformerModelInterface)
        mock_create_and_load.return_value = mock_model
        
        result = create_dialogpt()
        
        mock_create_and_load.assert_called_once()
        call_args = mock_create_and_load.call_args
        
        assert call_args[0][0] == "dialogpt"
        config = call_args[0][1]
        assert config.name == "microsoft/DialoGPT-large"
        assert config.use_quantization == False  # DialoGPT default

    @patch.object(ModelFactory, 'create_and_load')
    def test_create_dialogpt_with_custom_params(self, mock_create_and_load):
        """Test create_dialogpt function with custom parameters"""
        mock_model = Mock(spec=TransformerModelInterface)
        mock_create_and_load.return_value = mock_model
        
        result = create_dialogpt(
            model_name="microsoft/DialoGPT-medium",
            device_map="cpu"
        )
        
        mock_create_and_load.assert_called_once()
        call_args = mock_create_and_load.call_args
        
        assert call_args[0][0] == "dialogpt"
        config = call_args[0][1]
        assert config.name == "microsoft/DialoGPT-medium"
        assert config.use_quantization == False  # Default for DialoGPT
        assert config.device_map == "cpu"


class TestModelFactoryEdgeCases:
    """Test edge cases and error conditions"""

    def test_create_with_none_config(self):
        """Test creating model with None config raises appropriate error"""
        with pytest.raises(ValueError, match="Model configuration cannot be None"):
            ModelFactory.create("mistral", None)

    def test_create_with_empty_model_type(self):
        """Test creating model with empty model type"""
        config = ModelConfig(name="test-model")
        
        with pytest.raises(ValueError):
            ModelFactory.create("", config)

    def test_create_and_load_propagates_load_errors(self):
        """Test that create_and_load propagates load errors"""
        config = ModelConfig(name="test-model")
        
        with patch.object(MistralModel, 'load', side_effect=RuntimeError("Load failed")):
            with pytest.raises(RuntimeError, match="Load failed"):
                ModelFactory.create_and_load("mistral", config)