"""
Test configuration and fixtures for the localRagChat project.

This file is automatically discovered by pytest and provides shared
configuration constants and fixtures for all test modules.
"""

import pytest
from unittest.mock import Mock


# Test Configuration Constants
class TestConfig:
    """Test configuration constants to avoid hardcoded values in tests"""
    
    # Memory configuration
    DEFAULT_ACTIVE_LIMIT = 5
    LARGE_ACTIVE_LIMIT = 10
    SMALL_ACTIVE_LIMIT = 3
    
    # Test data
    TEST_SESSION_ID = "test-session-123"
    TEST_CONVERSATION_ID = "test-conversation-456"
    
    # Message roles
    USER_ROLE = "user"
    CHATBOT_ROLE = "chatbot"
    
    # Test messages
    SAMPLE_MESSAGES = [
        {"role": USER_ROLE, "content": "Hello"},
        {"role": CHATBOT_ROLE, "content": "Hi there!"},
        {"role": USER_ROLE, "content": "How are you?"},
        {"role": CHATBOT_ROLE, "content": "I'm doing well, thanks!"},
    ]


# Shared Fixtures
@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    mock_store.store_conversation.return_value = TestConfig.TEST_CONVERSATION_ID
    return mock_store


@pytest.fixture
def test_config():
    """Provide test configuration constants"""
    return TestConfig


# Memory-specific fixtures
@pytest.fixture
def default_memory_config():
    """Default memory configuration for tests"""
    return {
        "active_limit": TestConfig.DEFAULT_ACTIVE_LIMIT,
        "vector_store": None
    }


@pytest.fixture
def large_memory_config():
    """Large memory configuration for tests"""
    return {
        "active_limit": TestConfig.LARGE_ACTIVE_LIMIT,
        "vector_store": None
    }