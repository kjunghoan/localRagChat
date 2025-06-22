"""
Conversation storage using ChromaDB for vector-based memory.
"""

import chromadb
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json


ConversationHistory = List[Dict[str, str]]


class ConversationStorage:
    """Handles conversation persistence and retrieval"""

    def __init__(self, db_path: str = "./data/chroma_db"):
        """Init ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Get or create the collection for conversations
        Singleton
        """

        try:
            return self.client.get_collection("conversations")
        except ValueError:
            return self.client.create_collection("conversations")

    def store_conversation(self, conversation: ConversationHistory) -> str:
        """
        Store a complete conversation and return its ID.

        Args:
            conversation: List of message dicts with 'role' and 'content'.

        Returns:
            str: Conversation ID (UUID string)
        """
        if not conversation:
            raise ValueError("Cannot store empty conversation")
        # generate uuid
        conversation_id = str(uuid.uuid4())
        # create text summary for embedding
        conversation_text = self._conversation_to_text(conversation)
        # create metadata
        metadata = {
            "id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(conversation),
            "conversation_json": json.dumps(conversation),
        }
        self.collection.add(
            documents=[conversation_text], ids=[conversation_id], metadatas=[metadata]
        )
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """
        Retrieve a conversation by it's ID.
        Args:
            conversation_id: UUID string of the conversation.
        Returns:
            Conversation history or non if not found.
        """
        try:
            result = self.collection.get(ids=[conversation_id])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            conversation_json = metadata["conversation_json"]
            return json.loads(conversation_json)

        except Exception:
            return None

    def _conversation_to_text(self, conversation: ConversationHistory) -> str:
        """
        Convert conversation history to a single text string for embedding.
        """
        text_parts = []

        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content", "")
            text_parts.append(f"{role}: {content}")

        return "\n".join(text_parts)

    def list_conversations(self, limit: int = 10) -> List[Dict]:
        """
        Liust recent conversations with basic metadata
        Args:
            limit: Maximum number of conversations to return.
        Returns:
            List of conversation metadata dictionaries.
        """
        # Returns all conversations
        results = self.collection.get()

        if not results["ids"]:
            return []

        conversations = []
        for i, conv_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            conversations.append({
                "id": conv_id,
                "timestamp": metadata["timestamp"],
                "message_count": metadata["message_count"],
            })

        conversations.sort(key=lambda x: x["timestamp"], reverse=True)
        return conversations[:limit]
