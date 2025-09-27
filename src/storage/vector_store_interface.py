"""
Abstract interface for vector storage backends.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

ConversationHistory = List[Dict[str, str]]


@dataclass
class VectorStoreConfig:
    """Configuration for vector store backends (ChromaDB, PostgreSQL, etc.)"""

    embedding_model: str
    version: str = "dev"

    # ChromaDB specific settings
    db_path: str = "./data/vector_store"
    base_collection: str = "conversations"

    # PostgreSQL specific settings
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_database: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_table: str = "conversations"

    # Common settings
    dimension: Optional[int] = None
    index_type: Optional[str] = None

    def __post_init__(self):
        clean_model = self.embedding_model.replace("/", "_").replace("-", "_").lower()
        self.collection_name = f"{self.base_collection}_{clean_model}_{self.version}"

        # Load PostgreSQL settings from environment
        self.postgres_host = self.postgres_host or os.getenv("POSTGRES_HOST")
        self.postgres_port = self.postgres_port or (int(os.getenv("POSTGRES_PORT")) if os.getenv("POSTGRES_PORT") else None)
        self.postgres_database = self.postgres_database or os.getenv("POSTGRES_DATABASE")
        self.postgres_user = self.postgres_user or os.getenv("POSTGRES_USER")
        self.postgres_password = self.postgres_password or os.getenv("POSTGRES_PASSWORD")

    @classmethod
    def for_model(cls, embedding_model: str, **kwargs) -> "VectorStoreConfig":
        return cls(embedding_model=embedding_model, **kwargs)


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector storage backends.

    This interface defines the contract that all vector stores must follow,
    enabling easy swapping between ChromaDB, FAISS, Pinecone, etc.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    @abstractmethod
    def store_conversation(self, conversation: ConversationHistory) -> str:
        """
        Store a complete conversation and return its ID.

        Args:
            conversation: List of message dicts with 'role' and 'content'

        Returns:
            str: Conversation ID (UUID string)
        """
        pass

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """
        Retrieve a conversation by its ID.

        Args:
            conversation_id: UUID string of the conversation

        Returns:
            Conversation history or None if not found
        """
        pass

    @abstractmethod
    def list_conversations(self, limit: int = 10) -> List[Dict]:
        """
        List recent conversations with basic metadata

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation metadata dictionaries
        """
        pass

    @abstractmethod
    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for conversations similar to query (for future use)

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of similar conversation metadata
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized"""
        pass

    def get_store_info(self) -> Dict:
        """Get basic information about the vector store"""
        return {
            "backend": self.__class__.__name__,
            "db_path": self.config.db_path,
            "collection": self.config.collection_name,
            "embedding_model": self.config.embedding_model,
            "version": self.config.version,
            "initialized": self.is_initialized,
        }
