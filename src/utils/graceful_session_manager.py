import atexit
import signal
import sys
from src.memory.rolling_chat_memory import RollingChatMemory
from src.utils.logger import create_logger


class GracefulSessionManager:
    """
    Handles graceful shutdown and signal catching for memory safety.
    No changes needed from original implementation.
    """

    def __init__(self, memory: RollingChatMemory):
        self.memory = memory
        self.setup_signal_handlers()
        self.setup_exit_handler()
        self.logger = create_logger("GracefulSessionManager")

    def setup_signal_handlers(self):
        """Setup signal handlers for Ctrl-C, Ctrl-D, etc."""
        try:
            # Catch Ctrl-C (SIGINT)
            signal.signal(signal.SIGINT, self.graceful_exit)

            # Catch termination signals
            signal.signal(signal.SIGTERM, self.graceful_exit)

            # Catch hangup (terminal close)
            signal.signal(signal.SIGHUP, self.graceful_exit)

        except ValueError:
            # Some signals might not be available on all platforms
            pass

    def setup_exit_handler(self):
        """Setup emergency exit handler as backup"""
        atexit.register(self.emergency_save)

    def graceful_exit(self, sig, frame):
        """Handle caught signals gracefully"""
        _ = frame  # Unused, but required by signal handler signature
        self.logger.info(f"Caught signal {sig}, saving session...")
        self.save_session()
        self.logger.success("Session saved! Goodbye!")
        sys.exit(0)

    def emergency_save(self):
        """Emergency save on any Python exit"""
        if self.memory.has_unsaved_data():
            self.logger.info("Emergency save triggered...")
            self.save_session()

    def save_session(self):
        """Save session with error handling"""
        try:
            self.memory.session_cleanup()
        except Exception as e:
            self.logger.warning(f"Save failed: {e}")
