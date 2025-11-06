import os
from typing import List, Dict
from dataclasses import dataclass

ConversationHistory = List[Dict[str, str]]


@dataclass
class VectorStoreConfig:
    embedding_model: str
    version: str = "dev"
    database_url: str | None = None
    postgres_table: str = "conversations"
    dimension: int | None = None
    index_type: str | None = None

    def __post_init__(self):
        if self.database_url is None:
            self.database_url = os.getenv("DATABASE_URL")
            if not self.database_url:
                raise ValueError("DATABASE_URL environment variable is required")

    @classmethod
    def for_model(cls, embedding_model: str, **kwargs) -> "VectorStoreConfig":
        return cls(embedding_model=embedding_model, **kwargs)
