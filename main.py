#!/usr/bin/env python3
from src.configs import AppConfig
from src.auth import authenticate
from src.core.app import App


def main():
    """
    Main entry point
    """
    # Load configs
    # config = AppConfig.default()
    config = AppConfig.development()
    # config = AppConfig.production()
    # config = AppConfig.debug_mode()

    authenticate()
    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
