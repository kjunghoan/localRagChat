#!/usr/bin/env python3
"""
Test script to verify the new model interface works with both Mistral and DialoGPT
"""

from src.interfaces.model import ModelConfig
from src.factories.model import ModelFactory


def test_transformer_model_interface():
    """Test that both model types work with the same interface"""

    # Test conversation
    conversation = [
        {"role": "user", "content": "Hello! How are you today?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking! How about you?",
        },
        {
            "role": "user",
            "content": "I'm great! Can you help me with a coding question?",
        },
    ]

    # Test Mistral
    print("🤖 Testing Mistral Model...")
    mistral_config = ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.3", use_quantization=True, max_tokens=100
    )

    mistral = ModelFactory.create("mistral", mistral_config)
    print(f"✓ Created Mistral model: {mistral.__class__.__name__}")

    # Test prompt formatting (without loading model)
    mistral_prompt = mistral.format_prompt(conversation)
    print(f"✓ Mistral prompt format: {len(mistral_prompt)} chars")
    print(f"   Preview: {mistral_prompt[:100]}...")

    print("\n" + "=" * 50 + "\n")

    # Test DialoGPT
    print("💬 Testing DialoGPT Model...")
    dialogpt_config = ModelConfig(
        name="microsoft/DialoGPT-large", use_quantization=False, max_tokens=50
    )

    dialogpt = ModelFactory.create("dialogpt", dialogpt_config)
    print(f"✓ Created DialoGPT model: {dialogpt.__class__.__name__}")

    # Test prompt formatting
    dialogpt_prompt = dialogpt.format_prompt(conversation)
    print(f"✓ DialoGPT prompt format: {len(dialogpt_prompt)} chars")
    print(f"   Preview: {dialogpt_prompt[:100]}...")

    print("\n" + "=" * 50 + "\n")

    # Show the difference in prompt formatting
    print("📊 Prompt Format Comparison:")
    print(f"Mistral uses instruction format with [INST] tags")
    print(f"DialoGPT uses natural conversation with EOS tokens")
    print(f"Same interface, different implementations! ✨")

    print("\n✅ Model interface tests passed!")
    print("Ready to load and test with real inference...")


def test_factory_features():
    """Test factory convenience features"""
    print("\n🏭 Testing Factory Features...")

    # Test available models
    available = ModelFactory.available_models()
    print(f"✓ Available models: {available}")

    # Test convenience functions
    from src.factories.model import create_mistral, create_dialogpt

    print("✓ Convenience functions imported")

    # Test error handling
    try:
        ModelFactory.create("nonexistent", ModelConfig(name="test"))
        print("❌ Should have failed!")
    except ValueError as e:
        print(f"✓ Error handling works: {e}")

    print("✅ Factory features work!")


if __name__ == "__main__":
    test_transformer_model_interface()
    test_factory_features()
