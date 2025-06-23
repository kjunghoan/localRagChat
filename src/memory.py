"""
Memory management system with VRAM → RAM → Disk spillover
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import signal
import sys
import atexit

ConversationHistory = List[Dict[str, str]]


@dataclass
class Message:
    """Individual message with metadata"""

    id: str
    role: str
    content: str
    timestamp: datetime
    session_id: str

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
        """Convert to simple dict for AI context"""
        return {"role": self.role, "content": self.content}


class ConversationalMemory:
    """
    Manages conversation memory with performance-based spillover
    """

    def __init__(self, vram_limit: int = 15, ram_limit: int = 100, storage=None):
        """
        Initialize memory manager

        Args:
            vram_limit: Max messages in active context (for AI performance)
            ram_limit: Max messages in RAM before spilling to disk
            storage: ConversationStorage instance for disk operations
        """
        self.vram_messages: List[Message] = []  # Active context window
        self.ram_messages: List[Message] = []  # Session overflow
        self.vram_limit = vram_limit
        self.ram_limit = ram_limit
        self.storage = storage
        self.session_id = str(uuid.uuid4())

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

        # Add to VRAM (active context)
        self.vram_messages.append(message)
        self._has_unsaved = True

        # Check for VRAM spillover
        if len(self.vram_messages) > self.vram_limit:
            # Move oldest VRAM message to RAM
            spilled = self.vram_messages.pop(0)
            self.ram_messages.append(spilled)

            # Check for RAM spillover
            if len(self.ram_messages) > self.ram_limit:
                # Move oldest RAM message to disk (if storage available)
                if self.storage:
                    disk_spill = self.ram_messages.pop(0)
                    self._store_message_to_disk(disk_spill)

        return message

    def get_ai_context(self) -> ConversationHistory:
        """
        Get current conversation context for AI
        Returns all messages in session (RAM + VRAM in chronological order)
        """
        # RAM contains older messages, VRAM contains newer messages
        all_messages = self.ram_messages + self.vram_messages
        return [msg.to_dict() for msg in all_messages]

    def get_optimized_context(
        self, max_messages: Optional[int] = None
    ) -> ConversationHistory:
        """
        Get performance-optimized context (prioritizes recent messages)

        Args:
            max_messages: Limit total messages returned

        Returns:
            Recent messages as conversation history
        """
        # Prioritize VRAM, then recent RAM
        recent_ram = (
            self.ram_messages[-10:]
            if len(self.ram_messages) > 10
            else self.ram_messages
        )
        context_messages = self.vram_messages + recent_ram

        if max_messages and len(context_messages) > max_messages:
            context_messages = context_messages[-max_messages:]

        return [msg.to_dict() for msg in context_messages]

    def _store_message_to_disk(self, message: Message) -> None:
        """Store individual message to disk as unprocessed"""
        try:
            # For now, we'll store individual messages
            # Later this could be optimized to batch writes
            conversation_data = [message.to_dict()]
            self.storage.store_conversation(conversation_data)
        except Exception as e:
            print(f"⚠️  Failed to store message to disk: {e}")

    def session_cleanup(self) -> None:
        """
        Save all remaining session data to disk on exit
        """
        if not self._has_unsaved or not self.storage:
            return

        try:
            # Save all remaining messages as a conversation
            all_session_messages = self.vram_messages + self.ram_messages

            if all_session_messages:
                conversation_data = [msg.to_dict() for msg in all_session_messages]
                conv_id = self.storage.store_conversation(conversation_data)
                print(f"💾 Session saved to disk: {conv_id[:8]}...")

                # Clear memory
                self.vram_messages.clear()
                self.ram_messages.clear()
                self._has_unsaved = False

        except Exception as e:
            print(f"⚠️  Failed to save session: {e}")

    def has_unsaved_data(self) -> bool:
        """Check if there's unsaved data in memory"""
        return self._has_unsaved and (self.vram_messages or self.ram_messages)

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        return {
            "vram_messages": len(self.vram_messages),
            "ram_messages": len(self.ram_messages),
            "vram_limit": self.vram_limit,
            "ram_limit": self.ram_limit,
            "total_messages": len(self.vram_messages) + len(self.ram_messages),
        }


class GracefulSessionManager:
    """
    Handles graceful shutdown and signal catching for memory safety
    """

    def __init__(self, memory: ConversationalMemory):
        self.memory = memory
        self.setup_signal_handlers()
        self.setup_exit_handler()

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
        print(f"\n💾 Caught signal {sig}, saving session...")
        self.save_session()
        print("✅ Session saved! Goodbye!")
        sys.exit(0)

    def emergency_save(self):
        """Emergency save on any Python exit"""
        if self.memory.has_unsaved_data():
            print("🚨 Emergency save triggered...")
            self.save_session()

    def save_session(self):
        """Save session with error handling"""
        try:
            self.memory.session_cleanup()
        except Exception as e:
            print(f"⚠️  Save failed: {e}")
