import pytest
from datetime import datetime
import uuid

from src.memory.rolling_chat_memory import RollingChatMemory, Message


class TestMessage:
    """Test the message dataclass and its methods"""

    def test_message_creation(self):
        """Test manual message creation"""
        msg_id = str(uuid.uuid4())
        timestamp = datetime.now()
        session_id = str(uuid.uuid4())

        message = Message(
            id=msg_id,
            role="user",
            content="Hello, world!",
            timestamp=timestamp,
            session_id=session_id,
        )

        assert message.id == msg_id
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.timestamp == timestamp
        assert message.session_id == session_id

    def test_message_create_classmethod(self):
        """Test Message.create() auto-generation"""
        session_id = str(uuid.uuid4())
        message = Message.create("chatbot", "Hi there!", session_id)

        assert message.role == "chatbot"
        assert message.content == "Hi there!"
        assert message.session_id == session_id
        assert isinstance(message.id, str)
        assert isinstance(message.timestamp, datetime)

    def test_message_to_dict(self):
        """Test conversion to simple dict format"""
        session_id = str(uuid.uuid4())
        message = Message.create("user", "Test message", session_id)

        result = message.to_dict()
        expected = {"role": "user", "content": "Test message"}

        assert result == expected


class TestRollingChatMemory:
    """Test the rolling chat memory system"""

    @pytest.fixture
    def memory(self, mock_vector_store, test_config):
        """Create a rolling chat memory instance for testing"""
        return RollingChatMemory(
            active_limit=test_config.DEFAULT_ACTIVE_LIMIT, 
            vector_store=mock_vector_store
        )

    def test_initialization(self, mock_vector_store, test_config):
        """Test memory initialization"""
        memory = RollingChatMemory(
            active_limit=test_config.LARGE_ACTIVE_LIMIT, 
            vector_store=mock_vector_store
        )

        assert memory.active_limit == test_config.LARGE_ACTIVE_LIMIT
        assert memory.vector_store == mock_vector_store
        assert len(memory.active_messages) == 0
        assert not memory.has_unsaved_data()

    def test_add_message(self, memory):
        """Test adding messages to memory"""
        msg1 = memory.add_message("user", "Hello")

        assert len(memory.active_messages) == 1
        assert memory.active_messages[0] == msg1
        assert msg1.role == "user"
        assert msg1.content == "Hello"
        assert memory.has_unsaved_data()

        msg2 = memory.add_message("chatbot", "Hi there!")

        assert len(memory.active_messages) == 2
        assert memory.active_messages[1] == msg2

    def test_memory_spillover(self, memory):
        """Test that old messages are discarded from working memory"""
        messages = []
        for i in range(7):
            msg = memory.add_message("user", f"Message {i}")
            messages.append(msg)
        assert len(memory.active_messages) == memory.active_limit

        active_contents = [msg.content for msg in memory.active_messages]
        # Should keep the last active_limit messages
        start_idx = 7 - memory.active_limit
        expected_contents = [f"Message {i}" for i in range(start_idx, 7)]
        assert active_contents == expected_contents

    def test_get_ai_context_full(self, memory):
        """Test getting full AI context"""
        memory.add_message("user", "Hello")
        memory.add_message("chatbot", "Hi there!")
        memory.add_message("user", "How are you?")

        context = memory.get_ai_context()

        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "chatbot", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        assert context == expected

    def test_get_ai_context_limited(self, memory):
        """Test getting limited AI context"""
        memory.add_message("user", "Message 1")
        memory.add_message("chatbot", "Response 1")
        memory.add_message("user", "Message 2")
        memory.add_message("chatbot", "Response 2")

        context = memory.get_ai_context(max_messages=2)

        expected = [
            {"role": "user", "content": "Message 2"},
            {"role": "chatbot", "content": "Response 2"},
        ]
        assert context == expected

    def test_get_ai_context_empty(self, memory):
        """Test getting context when no messages exist"""
        context = memory.get_ai_context()
        assert context == []

    def test_session_cleanup_with_messages(self, memory):
        """Test session cleanup saves messages to vector store"""
        # Add some messages
        memory.add_message("user", "Hello")
        memory.add_message("chatbot", "Hi!")

        assert memory.has_unsaved_data()

        # Cleanup session
        memory.session_cleanup()

        # Verify vector store was called
        memory.vector_store.store_conversation.assert_called_once()

        # Check the conversation data passed to store
        call_args = memory.vector_store.store_conversation.call_args[0][0]
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "chatbot", "content": "Hi!"},
        ]
        assert call_args == expected

        # Verify memory is cleared
        assert len(memory.active_messages) == 0
        assert not memory.has_unsaved_data()

    def test_session_cleanup_no_messages(self, memory):
        """Test session cleanup with no messages doesn't call vector store"""
        memory.session_cleanup()

        # Vector store should not be called
        memory.vector_store.store_conversation.assert_not_called()

    def test_session_cleanup_no_vector_store(self, test_config):
        """Test session cleanup without vector store"""
        memory = RollingChatMemory(
            active_limit=test_config.DEFAULT_ACTIVE_LIMIT, 
            vector_store=None
        )
        memory.add_message("user", "Hello")

        # Should not raise exception
        memory.session_cleanup()
        assert len(memory.active_messages) == 1  # Messages remain

    def test_session_cleanup_with_error(self, memory):
        """Test session cleanup handles vector store errors gracefully"""
        memory.add_message("user", "Hello")

        # Make vector store raise an exception
        memory.vector_store.store_conversation.side_effect = Exception("Storage error")

        # Should not raise exception, just log warning
        memory.session_cleanup()

        # Messages should remain since save failed
        assert len(memory.active_messages) == 1
        assert memory.has_unsaved_data()

    def test_has_unsaved_data(self, memory):
        """Test unsaved data detection"""
        # Initially no unsaved data
        assert not memory.has_unsaved_data()

        # Add message creates unsaved data
        memory.add_message("user", "Hello")
        assert memory.has_unsaved_data()

        # Cleanup removes unsaved data flag
        memory.session_cleanup()
        assert not memory.has_unsaved_data()

    def test_get_memory_stats(self, memory):
        """Test memory statistics"""
        # Empty memory
        stats = memory.get_memory_stats()
        assert stats == {"active_messages": 0, "active_limit": memory.active_limit, "total_messages": 0}

        # Add some messages
        memory.add_message("user", "Hello")
        memory.add_message("chatbot", "Hi!")

        stats = memory.get_memory_stats()
        assert stats == {"active_messages": 2, "active_limit": memory.active_limit, "total_messages": 2}

    def test_session_id_consistency(self, memory):
        """Test that all messages in a session have the same session_id"""
        memory.add_message("user", "Message 1")
        memory.add_message("chatbot", "Response 1")
        memory.add_message("user", "Message 2")

        session_ids = [msg.session_id for msg in memory.active_messages]

        # All should have the same session ID
        assert len(set(session_ids)) == 1
        assert session_ids[0] == memory.session_id
