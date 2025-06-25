"""
Rolling chat memory system with RAM -> Disk spillover for performance optimization.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
from src.utils.logger import create_logger

ConversationHistory = List[Dict[str, str]]


@dataclass
class Message:
    """Individual message with metadata"""

    id: str
    role: str
    content: str
    timestamp: datetime
    session_id: str # UUID

    @classmethod
    def create(cls, role: str, content: str, session_id: str) -> "Message":
        """Create a new message with auto-generated ID and timestamp"""
        return cls(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            session_id=session_id,
        )

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to simple dict
        """
        return {"role": self.role, "content": self.content}


class RollingChatMemory:
    """
    Manages conversation memory with performance-based spillover.
    RAM → "fast" storage in the form of Session persistence, to vector store on exit.
    """

    def __init__(self, active_limit: int = 100, vector_store=None):
        """
        Initialize memory manager

        Args:
            active_limit: Max messages in RAM before oldest are discarded
            vector_store: ConversationVectorStore instance for session persistence
        """
        self.active_messages: List[Message] = []  # Active performance tier
        self.active_limit = active_limit
        self.vector_store = vector_store
        self.session_id = str(uuid.uuid4())
        self.logger = create_logger("RollingChatMemory")

        # Track if we have unsaved data
        self._has_unsaved = False

    def add_message(self, role: str, content: str) -> Message:
        """
        Add a new message to memory with automatic spillover

        Args:
            role: 'user' or 'chatbot'
            content: Message content

        Returns:
            Created Message object
        """
        message = Message.create(role, content, self.session_id)

        # Add to active memory tier
        self.active_messages.append(message)
        self._has_unsaved = True

        # Check for active memory overflow
        if len(self.active_messages) > self.active_limit:
            self.active_messages.pop(0)

        return message


    def get_ai_context(self, max_messages: Optional[int] = None) -> ConversationHistory:
        """
        Get current conversation context for AI.
        Returns recent messages from session (messages in ram).

        Args:
            max_messages: Override to limit context size (uses config default if None)
        """
        if max_messages is not None:
            recent = self.active_messages[-max_messages:]
        else:
            recent = self.active_messages
        return [msg.to_dict() for msg in recent]

    def session_cleanup(self) -> None:
        """
        Save all remaining session data to vector store on exit.
        This is the only integration point with persistent storage.
        """
        if not self._has_unsaved or not self.vector_store:
            return

        try:
            if self.active_messages:
                conversation_data = [msg.to_dict() for msg in self.active_messages]
                conv_id = self.vector_store.store_conversation(conversation_data)
                self.logger.storage(f"Session saved to vector store: {conv_id[:8]}...")

                # Clear memory
                self.active_messages.clear()
                self._has_unsaved = False

        except Exception as e:
            self.logger.warning(f"Failed to save session: {e}")

    def has_unsaved_data(self) -> bool:
        """Check if there's unsaved data in memory"""
        return self._has_unsaved and (len(self.active_messages) > 0)

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        return {
            "active_messages": len(self.active_messages),
            "active_limit": self.active_limit,
            "total_messages": len(self.active_messages)
        }
