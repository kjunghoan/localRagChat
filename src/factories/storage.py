"""
Factory for creating vector storage instances.
"""

from typing import Dict, Type, List
from ..storage.vector_store_interface import VectorStoreInterface, VectorStoreConfig
from ..storage.chromadb_store import ChromaDBStore


class StorageFactory:
    """Factory for creating vector storage instances based on configuration"""

    # Registry of available storage backends
    _storage_registry: Dict[str, Type[VectorStoreInterface]] = {
        "chromadb": ChromaDBStore,
        # "faiss": FaissStore,        # Future implementation
    }

    @classmethod
    def create(
        cls, storage_type: str, config: VectorStoreConfig
    ) -> VectorStoreInterface:
        """
        Create a vector storage instance based on type and configuration.

        Args:
            storage_type: Type of storage ("chromadb", "faiss", etc.)
            config: VectorStoreConfig with storage-specific settings

        Returns:
            Initialized storage instance

        Raises:
            ValueError: If storage_type is not supported
        """
        if storage_type not in cls._storage_registry:
            available = ", ".join(cls.available_storage_types())
            raise ValueError(
                f"Unknown storage type '{storage_type}'. Available: {available}"
            )

        storage_class = cls._storage_registry[storage_type]
        return storage_class(config)

    @classmethod
    def available_storage_types(cls) -> List[str]:
        """Get list of available storage types"""
        return list(cls._storage_registry.keys())

    @classmethod
    def register_storage(
        cls, storage_type: str, storage_class: Type[VectorStoreInterface]
    ) -> None:
        # TODO: for v2 or v3 possibly allow for adaptation from vector store to object
        # store for conversation content
        """
        Register a new storage type (for extensibility).

        Args:
            storage_type: String identifier for the storage backend
            storage_class: Class that implements VectorStoreInterface
        """
        cls._storage_registry[storage_type] = storage_class


# Convenience function for common storage creation
def create_chromadb_store(
    embedding_model: str, db_path: str = "./data/vector_stores", **kwargs
) -> VectorStoreInterface:
    """Create a ChromaDB store with sensible defaults"""
    config = VectorStoreConfig(
        embedding_model=embedding_model, db_path=db_path, **kwargs
    )
    return StorageFactory.create("chromadb", config)
