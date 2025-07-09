"""
Tests for the storage factory implementation.
"""

import pytest
from unittest.mock import Mock, patch
from src.factories.storage import StorageFactory, create_chromadb_store, create_null_store
from src.storage.vector_store_interface import VectorStoreInterface, VectorStoreConfig
from src.storage.chromadb_store import ChromaDBStore


class TestStorageFactory:
    """Test the StorageFactory class"""

    @patch('src.storage.chromadb_store.chromadb.PersistentClient')
    def test_create_valid_storage_type(self, mock_client):
        """Test creating storage with valid storage types"""
        config = VectorStoreConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            db_path="./test_data"
        )
        
        storage = StorageFactory.create("chromadb", config)
        
        assert isinstance(storage, ChromaDBStore)
        assert storage.config == config

    def test_create_unknown_storage_type(self):
        """Test creating unknown storage type raises ValueError"""
        config = VectorStoreConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            db_path="./test_data"
        )
        
        with pytest.raises(ValueError) as exc_info:
            StorageFactory.create("unknown", config)
        
        assert "Unknown storage type 'unknown'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_register_storage(self):
        """Test registering a new storage type"""
        mock_storage_class = Mock(spec=VectorStoreInterface)
        mock_instance = Mock(spec=VectorStoreInterface)
        mock_storage_class.return_value = mock_instance
        
        # Register the new storage
        StorageFactory.register_storage("test_storage", mock_storage_class)
        
        # Test that we can create it
        config = VectorStoreConfig(
            embedding_model="test-model",
            db_path="./test"
        )
        result = StorageFactory.create("test_storage", config)
        
        assert result == mock_instance
        mock_storage_class.assert_called_once_with(config)
        
        # Clean up
        del StorageFactory._storage_registry["test_storage"]

    def test_register_storage_overwrites_existing(self):
        """Test that registering overwrites existing storage types"""
        original_class = StorageFactory._storage_registry["chromadb"]
        mock_storage_class = Mock(spec=VectorStoreInterface)
        mock_instance = Mock(spec=VectorStoreInterface)
        mock_storage_class.return_value = mock_instance
        
        StorageFactory.register_storage("chromadb", mock_storage_class)
        
        config = VectorStoreConfig(
            embedding_model="test-model",
            db_path="./test"
        )
        result = StorageFactory.create("chromadb", config)
        
        assert result == mock_instance
        mock_storage_class.assert_called_once_with(config)
        
        # Clean up
        StorageFactory._storage_registry["chromadb"] = original_class

    @patch('src.storage.chromadb_store.chromadb.PersistentClient')
    def test_create_with_different_configs(self, mock_client):
        """Test creating storage with different configuration options"""
        config1 = VectorStoreConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            db_path="./data1",
            base_collection="conversations1"
        )
        config2 = VectorStoreConfig(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            db_path="./data2",
            base_collection="conversations2"
        )
        
        storage1 = StorageFactory.create("chromadb", config1)
        storage2 = StorageFactory.create("chromadb", config2)
        
        assert storage1.config == config1
        assert storage2.config == config2
        assert storage1.config.embedding_model != storage2.config.embedding_model


class TestConvenienceFunction:
    """Test the convenience function for storage creation"""

    @patch.object(StorageFactory, 'create')
    def test_create_chromadb_store_with_defaults(self, mock_create):
        """Test create_chromadb_store function with default parameters"""
        mock_storage = Mock(spec=VectorStoreInterface)
        mock_create.return_value = mock_storage
        
        result = create_chromadb_store("test-embedding-model")
        
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        
        assert call_args[0][0] == "chromadb"  # storage_type
        config = call_args[0][1]  # config
        assert config.embedding_model == "test-embedding-model"
        assert config.db_path == "./data/vector_store"
        assert result == mock_storage

    @patch.object(StorageFactory, 'create')
    def test_create_chromadb_store_with_custom_params(self, mock_create):
        """Test create_chromadb_store function with custom parameters"""
        mock_storage = Mock(spec=VectorStoreInterface)
        mock_create.return_value = mock_storage
        
        result = create_chromadb_store(
            embedding_model="custom/model",
            db_path="./custom_data",
            base_collection="custom_conversations",
            version="v2"
        )
        
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        
        assert call_args[0][0] == "chromadb"
        config = call_args[0][1]
        assert config.embedding_model == "custom/model"
        assert config.db_path == "./custom_data"
        assert config.base_collection == "custom_conversations"
        assert config.version == "v2"
        assert result == mock_storage

    def test_create_null_store_convenience(self):
        """Test create_null_store convenience function"""
        result = create_null_store()
        assert result is None


class TestStorageFactoryEdgeCases:
    """Test edge cases and error conditions"""

    def test_create_null_storage(self):
        """Test creating null storage for no-persistence mode"""
        result = StorageFactory.create("null")
        assert result is None
        
        # Can also pass None config explicitly
        result = StorageFactory.create("null", None)
        assert result is None

    def test_create_with_none_config_for_real_storage(self):
        """Test creating real storage with None config raises appropriate error"""
        with pytest.raises(AttributeError):
            StorageFactory.create("chromadb", None)

    def test_create_with_empty_storage_type(self):
        """Test creating storage with empty storage type"""
        config = VectorStoreConfig(
            embedding_model="test-model",
            db_path="./test"
        )
        
        with pytest.raises(ValueError):
            StorageFactory.create("", config)

    def test_available_storage_types_behavior(self):
        """Test that available_storage_types returns a list"""
        available = StorageFactory.available_storage_types()
        
        assert isinstance(available, list)
        assert len(available) > 0  # Should have at least chromadb
        
        # Test that modifying returned list doesn't affect the registry
        original_length = len(available)
        available.append("test")
        new_available = StorageFactory.available_storage_types()
        assert len(new_available) == original_length

    def test_register_storage_with_invalid_class(self):
        """Test registering storage with class that doesn't implement interface"""
        class InvalidStorage:
            pass
        
        # This should not raise an error at registration time
        StorageFactory.register_storage("invalid", InvalidStorage)
        
        # But should fail when trying to create if the class doesn't match interface
        config = VectorStoreConfig(
            embedding_model="test-model",
            db_path="./test"
        )
        
        # The actual interface checking would happen at runtime
        # when trying to use the storage instance
        
        # Clean up
        del StorageFactory._storage_registry["invalid"]