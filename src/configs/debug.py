"""
Debug and logging configuration.
"""

from dataclasses import dataclass


@dataclass
class DebugConfig:
    """Debug and logging configuration"""

    # Core debug settings
    enabled: bool = False

    # Model loading debug
    verbose_loading: bool = False

    # Generation debug
    show_token_counts: bool = False
    show_full_responses: bool = False
