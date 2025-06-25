"""
Chat behavior and generation logic
"""

from dataclasses import dataclass

@dataclass
class ChatConfig:
    """Chat behavior configuration"""

    max_tokens: int = 300 # Maximum number of tokens to generate
    max_length: int = 1024 # Maximum length of input text
    temperature: float = 0.7 # Sampling temperature for generation
    do_sample: bool = True # Whether to use sampling or greedy decoding

    # Memory adn context management
    context_messages: int = 6 # Number of messages to keep in context
    active_limit: int = 100 # Maximum number of messages in active memory
