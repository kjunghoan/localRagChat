import json
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime

import torch
import psycopg
from psycopg.rows import dict_row
from psycopg import sql
from sentence_transformers import SentenceTransformer

from src.utils.logger import create_logger
from .vector_store_interface import VectorStoreConfig, ConversationHistory


class PgVectorStore:
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.logger = create_logger("PgVectorStore")
        self._conn = None
        self._embedding_model = None
        self._initialize()

    def _is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def _initialize(self):
        try:
            self._conn = psycopg.connect(self.config.database_url, row_factory=dict_row)
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
                f"Connected to PostgreSQL: {self.config.database_url.split('@')[1]}"
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
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id UUID PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({}),
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        message_count INTEGER,
                        conversation_json JSONB
                    );
                """
                ).format(
                    sql.Identifier(self.config.postgres_table),
                    sql.Literal(embedding_dim),
                )
            )

            # Create index for vector similarity search
            index_name = f"{self.config.postgres_table}_embedding_idx"
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {}
                    ON {}
                    USING hnsw (embedding vector_cosine_ops);
                """
                ).format(
                    sql.Identifier(index_name),
                    sql.Identifier(self.config.postgres_table),
                )
            )

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
                sql.SQL(
                    """
                    INSERT INTO {}
                    (id, content, embedding, metadata, message_count, conversation_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                ).format(sql.Identifier(self.config.postgres_table)),
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
                    sql.SQL(
                        """
                        SELECT conversation_json
                        FROM {}
                        WHERE id = %s
                    """
                    ).format(sql.Identifier(self.config.postgres_table)),
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
                    sql.SQL(
                        """
                        SELECT id, created_at, message_count, metadata
                        FROM {}
                        ORDER BY created_at DESC
                        LIMIT %s
                    """
                    ).format(sql.Identifier(self.config.postgres_table)),
                    (limit,),
                )

                conversations = []
                for row in cur.fetchall():
                    conversations.append(
                        {
                            "id": str(row["id"]),
                            "stored_at": (
                                row["created_at"].isoformat()
                                if row["created_at"]
                                else ""
                            ),
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
                    sql.SQL(
                        """
                        SELECT
                            id,
                            created_at,
                            message_count,
                            metadata,
                            1 - (embedding <=> %s) as similarity_score
                        FROM {}
                        ORDER BY embedding <=> %s
                        LIMIT %s
                    """
                    ).format(sql.Identifier(self.config.postgres_table)),
                    (query_embedding, query_embedding, limit),
                )

                similar_conversations = []
                for row in cur.fetchall():
                    similar_conversations.append(
                        {
                            "id": str(row["id"]),
                            "stored_at": (
                                row["created_at"].isoformat()
                                if row["created_at"]
                                else ""
                            ),
                            "message_count": row["message_count"],
                            "similarity_score": (
                                float(row["similarity_score"])
                                if row["similarity_score"]
                                else None
                            ),
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
                cur.execute(
                    sql.SQL(
                        """
                        SELECT
                            COUNT(*) as total_conversations,
                            AVG(message_count) as avg_messages_per_conversation,
                            MAX(created_at) as latest_conversation,
                            MIN(created_at) as oldest_conversation
                        FROM {}
                    """
                    ).format(sql.Identifier(self.config.postgres_table))
                )

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
