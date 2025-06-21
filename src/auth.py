from huggingface_hub import login
import os
from dotenv import load_dotenv


def authenticate() -> None:
    """Handle Hugging Face authentication"""
    load_dotenv()
    token: str | None = os.getenv("HF_TOKEN")

    if token:
        print("Logging in with token...")
        login(token)
    else:
        print("No token found, using cached login...")
        login()
