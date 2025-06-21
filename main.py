from src.auth import authenticate
from src.models import load_model_and_tokenizer
from src.chat import chat_loop
from dataclasses import dataclass

@dataclass
class ChatConfig:
    """
    Configuration for chat settings
    """
    max_tokens: int = 50
    max_length: int = 512

def main():
    """
    Main entry point
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    authenticate()
    model, tokenizer = load_model_and_tokenizer(model_name)

    chat_config = ChatConfig(max_tokens=50, max_length=512)
    chat_loop(model, tokenizer, chat_config)

if __name__ == "__main__":
    main()
