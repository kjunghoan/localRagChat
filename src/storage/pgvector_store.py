"""
PostgreSQL + pgvector implementation of vector storage interface.
"""

# import os
import json
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime

import torch
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

from src.utils.logger import create_logger
from .vector_store_interface import (
    VectorStoreInterface,
    VectorStoreConfig,
    ConversationHistory,
)


class PgVectorStore(VectorStoreInterface):
    """PostgreSQL + pgvector implementation of VectorStoreInterface"""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.logger = create_logger("PgVectorStore")
        self._conn = None
        self._embedding_model = None
        self._initialize()

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration"""
        return torch.cuda.is_available()

    def _initialize(self):
        """Initialize PostgreSQL connection and embedding model"""
        try:
            conn_str = f"postgresql://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}"

            self._conn = psycopg.connect(conn_str, row_factory=dict_row)
            self._conn.autocommit = True

            # Load embedding model (use CUDA if available for better performance)
            device = "cuda" if self._is_cuda_available() else "cpu"
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model, device=device
            )
            self.logger.info(f"Loaded embedding model on device: {device}")

            # Setup database schema
            self._setup_schema()

            self.logger.info(
                f"Connected to PostgreSQL at {self.config.postgres_host}:{self.config.postgres_port}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize PgVectorStore: {e}")
            raise

    def _setup_schema(self):
        """Setup database schema with pgvector extension and tables"""
        with self._conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Get embedding dimension
            sample_embedding = self._embedding_model.encode("test")
            embedding_dim = len(sample_embedding)

            # Create conversations table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.postgres_table} (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({embedding_dim}),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    message_count INTEGER,
                    conversation_json JSONB
                );
            """)

            # Create index for vector similarity search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.config.postgres_table}_embedding_idx
                ON {self.config.postgres_table}
                USING hnsw (embedding vector_cosine_ops);
            """)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformer"""
        embedding = self._embedding_model.encode(text)
        return embedding.tolist()

    def store_conversation(self, conversation: ConversationHistory) -> str:
        """Store a complete conversation and return its ID"""
        if not conversation:
            raise ValueError("Cannot store empty conversation")

        conversation_id = str(uuid.uuid4())
        conversation_text = self._conversation_to_text(conversation)
        embedding = self._generate_embedding(conversation_text)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(conversation),
        }

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.config.postgres_table}
                (id, content, embedding, metadata, message_count, conversation_json)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (
                    conversation_id,
                    conversation_text,
                    embedding,
                    json.dumps(metadata),
                    len(conversation),
                    json.dumps(conversation),
                ),
            )

        self.logger.info(
            f"Stored conversation {conversation_id} with {len(conversation)} messages"
        )
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Retrieve a conversation by its ID"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT conversation_json
                    FROM {self.config.postgres_table}
                    WHERE id = %s
                """,
                    (conversation_id,),
                )

                result = cur.fetchone()
                if result and result["conversation_json"]:
                    return result["conversation_json"]
                return None

        except Exception as e:
            self.logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None

    def list_conversations(self, limit: int = 10) -> List[Dict]:
        """List recent conversations with basic metadata"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, created_at, message_count, metadata
                    FROM {self.config.postgres_table}
                    ORDER BY created_at DESC
                    LIMIT %s
                """,
                    (limit,),
                )

                conversations = []
                for row in cur.fetchall():
                    conversations.append(
                        {
                            "id": str(row["id"]),
                            "stored_at": row["created_at"].isoformat()
                            if row["created_at"]
                            else "",
                            "message_count": row["message_count"],
                            "metadata": row["metadata"],
                        }
                    )

                return conversations

        except Exception as e:
            self.logger.error(f"Failed to list conversations: {e}")
            return []

    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for conversations similar to query using vector similarity"""
        try:
            query_embedding = self._generate_embedding(query)

            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        created_at,
                        message_count,
                        metadata,
                        1 - (embedding <=> %s) as similarity_score
                    FROM {self.config.postgres_table}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """,
                    (query_embedding, query_embedding, limit),
                )

                similar_conversations = []
                for row in cur.fetchall():
                    similar_conversations.append(
                        {
                            "id": str(row["id"]),
                            "stored_at": row["created_at"].isoformat()
                            if row["created_at"]
                            else "",
                            "message_count": row["message_count"],
                            "similarity_score": float(row["similarity_score"])
                            if row["similarity_score"]
                            else None,
                            "metadata": row["metadata"],
                        }
                    )

                return similar_conversations

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    @property
    def is_initialized(self) -> bool:
        """Check if PostgreSQL connection is properly initialized"""
        try:
            return self._conn is not None and not self._conn.closed
        except:
            return False

    def _conversation_to_text(self, conversation: ConversationHistory) -> str:
        """Convert conversation history to a single text string for embedding"""
        text_parts = []
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content", "")
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_conversations,
                        AVG(message_count) as avg_messages_per_conversation,
                        MAX(created_at) as latest_conversation,
                        MIN(created_at) as oldest_conversation
                    FROM {self.config.postgres_table}
                """)

                result = cur.fetchone()
                return dict(result) if result else {}

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self.logger.info("Closed PostgreSQL connection")

