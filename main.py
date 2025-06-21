#!/usr/bin/env python3
from src.config import AppConfig
from src.auth import authenticate
from src.models import load_model_and_tokenizer
from src.chat import chat_loop


def main():
    """
    Main entry point
    """
    # config = AppConfig.default()
    # config = AppConfig.development()
    config = AppConfig.production()
    # config = AppConfig.debug_mode()

    # Show what we're using
    if config.debug.enabled:
        print("🤖 Local RAG Chat starting with:")
        print(f"   Model: {config.model.name}")
        print(
            f"   Quantization: {'Enabled' if config.model.use_quantization else 'Disabled'}"
        )
        print(f"   Max tokens: {config.chat.max_tokens}")
        print(f"   Context messages: {config.chat.context_messages}")
        print(f"   Debug: {'On' if config.debug.enabled else 'Off'}")
        print()

    # Setup
    config.ensure_directories()
    authenticate()

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        config.model.name, use_quantization=config.model.use_quantization
    )

    # Start chat with individual config components
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        chat_config=config.chat,
        debug=config.debug.enabled,
        context_messages=config.chat.context_messages,
    )


if __name__ == "__main__":
    main()
