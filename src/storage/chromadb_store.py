"""
ChromaDB implementation of vector storage interface.
"""

import chromadb
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json
from src.utils.logger import create_logger

from .vector_store_interface import (
    VectorStoreInterface,
    VectorStoreConfig,
    ConversationHistory,
)


class ChromaDBStore(VectorStoreInterface):
    """ChromaDB implementation of VectorStoreInterface"""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.logger = create_logger("ChromaDBStore")
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(path=self.config.db_path)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create the collection for conversations"""
        try:
            return self.client.get_collection(self.config.collection_name)
        except Exception:
            return self.client.create_collection(self.config.collection_name)

    def store_conversation(self, conversation: ConversationHistory) -> str:
        """Store a complete conversation in it's versioned db and return its ID"""
        if not conversation:
            raise ValueError("Cannot store empty conversation")

        conversation_id = str(uuid.uuid4())
        conversation_text = self._conversation_to_text(conversation)

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
        """Retrieve a conversation by its ID"""
        try:
            result = self.collection.get(ids=[conversation_id])

            if not result["ids"] or not result["metadatas"]:
                return None

            metadata = result["metadatas"][0]
            if metadata is None:
                return None

            conversation_json = metadata["conversation_json"]
            if not isinstance(conversation_json, str):
                return None

            return json.loads(conversation_json)

        except Exception:
            return None

    def list_conversations(self, limit: int = 10) -> List[Dict]:
        """List recent conversations with basic metadata"""
        results = self.collection.get()

        if not results["ids"] or not results["metadatas"]:
            return []

        conversations = []
        for i, conv_id in enumerate(results["ids"]):
            if i >= len(results["metadatas"]) or results["metadatas"][i] is None:
                continue

            metadata = results["metadatas"][i]
            conversations.append({
                "id": conv_id,
                "stored_at": metadata.get("stored_at", ""),
                "message_count": metadata.get("message_count"),
            })

        conversations.sort(key=lambda x: x["stored_at"], reverse=True)
        return conversations[:limit]

    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for conversations similar to query (ChromaDB implementation)"""
        try:
            results = self.collection.query(query_texts=[query], n_results=limit)

            similar_conversations = []
            if not results["ids"] or not results["ids"][0]:
                return similar_conversations
            if not results["metadatas"] or not results["metadatas"][0]:
                return similar_conversations

            ids_list = results["ids"][0]
            metadatas_list = results["metadatas"][0]

            distances = results.get("distances")
            distances_list = distances[0] if distances and len(distances) > 0 else None

            for i, conv_id in enumerate(ids_list):

                if i >= len(metadatas_list) or metadatas_list[i] is None:
                    continue

                metadata = metadatas_list[i]
                distance = distances_list[i] if distances_list and i < len(distances_list) else None

                similar_conversations.append({
                    "id": conv_id,
                    "stored_at": metadata.get("stored_at", ""),
                    "message_count": metadata.get("message_count", 0),
                    "similarity_score": 1 - distance if distance else None,
                })

            return similar_conversations

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    @property
    def is_initialized(self) -> bool:
        """Check if ChromaDB is properly initialized"""
        return self.client is not None and self.collection is not None

    def _conversation_to_text(self, conversation: ConversationHistory) -> str:
        """Convert conversation history to a single text string for embedding"""
        text_parts = []
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content", "")
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)
