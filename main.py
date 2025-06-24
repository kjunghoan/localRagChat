#!/usr/bin/env python3
from src.config import AppConfig
from src.auth import authenticate


def main():
    """
    Main entry point
    """
    # Load configs
    # config = AppConfig.default()
    # config = AppConfig.development()
    # config = AppConfig.production()
    config = AppConfig.debug_mode()

    # Setup Auth
    authenticate()

    # Create and run app
    # TODO: Replace with proper ChatApp when ready
    from src.core.temp_app import TempApp

    app = TempApp(config)
    app.run()


if __name__ == "__main__":
    main()
