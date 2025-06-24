from enum import Enum
from datetime import datetime


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class Logger:
    """
    Custom logger that is compatible with python's logging interface
    Can be easily replace with standard logging via find/replace if needed
    """

    def __init__(self, name: str = "LocalRAGChat", debug_enabled: bool = False):
        self.name = name
        self.debug_enabled = debug_enabled
        self.min_level = LogLevel.DEBUG if debug_enabled else LogLevel.INFO

    def _log(self, level: LogLevel, message: str, emoji: str = "") -> None:
        """Private method to handle logging"""
        if level.value < self.min_level.value:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {emoji}" if emoji else f"[{timestamp}]"
        print(f"{prefix} {message}")

    def debug(self, message: str) -> None:
        """Debug messages - only shown when debug enabled"""
        self._log(LogLevel.DEBUG, message, "🔍")

    def info(self, message: str) -> None:
        """General information messages"""
        self._log(LogLevel.INFO, message, "ℹ️")

    def warning(self, message: str) -> None:
        """Warning messages"""
        self._log(LogLevel.WARNING, message, "⚠️")

    def error(self, message: str) -> None:
        """Error messages"""
        self._log(LogLevel.ERROR, message, "❌")

    # Convenience methods (delegate to standard methods for compatibility)
    def success(self, message: str) -> None:
        """Success/completion messages"""
        self.info(f"✅ {message}")

    def step(self, message: str) -> None:
        """Process step messages"""
        self.info(f"🔧 {message}")

    def memory(self, message: str) -> None:
        """Memory-related messages"""
        self.debug(f"🧠 {message}")

    def model(self, message: str) -> None:
        """Model-related messages"""
        self.info(f"🤖 {message}")

    def storage(self, message: str) -> None:
        """Storage-related messages"""
        self.info(f"💾 {message}")

    def factory(self, message: str) -> None:
        """Factory creation messages"""
        self.info(f"🏭 {message}")

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging"""
        self.debug_enabled = enabled
        self.min_level = LogLevel.DEBUG if enabled else LogLevel.INFO


def create_logger(name: str = "LocalRAGChat", debug: bool = False) -> Logger:
    """Create a logger instance"""
    return Logger(name, debug)
