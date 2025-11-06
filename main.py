#!/usr/bin/env python3
from dotenv import load_dotenv
from src.config import Config
from src.auth import authenticate
from src.core.app import App


def main():
    load_dotenv()
    config = Config.from_env()
    authenticate()
    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
