# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project is meant to be a distilled module that will go into a larger project, so the
versioning is not strictly semantic but rather a reflection of the development process.

## [Unreleased]

## [0.1.1] - Test Infrastructure & Cleanup - 2025-07-09

### Added
- Comprehensive test configuration system with `tests/conftest.py`
- Centralized test constants and fixtures for maintainable testing
- Integration test suite for rolling chat memory system (`tests/integration/test_app_integration.py`)
- Comprehensive factory testing (`tests/factories/test_model.py`, `tests/factories/test_storage.py`)
- Null storage support (`StorageFactory.create("null")`) for no-persistence mode
- `create_null_store()` convenience function for testing and development
- Configuration validation in model and storage factories

### Fixed
- Circular import issue between utils and memory modules

### Changed
- Standardized vector store directory path from `vector_stores` to `vector_store`
- Updated import structure to avoid circular dependencies
- Storage factory now supports optional config parameter for null storage
- Enhanced error messages in factories to include available options

### Removed
- `debug_storage.py` utility script
- `view_conversations.py` utility script
- Removed `GracefulSessionManager` from `utils/__init__.py` exports

## [0.1.0] - Modular Architecture Foundation - 2025-06-27

- Modular configuration system with separate configs for models, chat, debug, and app settings
- SupportedModel enum for type-safe model selection and management
- Factory pattern implementation for models and storage with dependency injection
- Rolling chat memory system with configurable active limits and automatic overflow handling
- Vector storage interface with ChromaDB implementation for conversation persistence
- Graceful session management with signal handling for safe exits
- Support for multiple AI models (Mistral 7B, DialoGPT Large/Medium) with unified interface
- Comprehensive installation documentation (INSTALL.md)
- Project README with architecture overview and roadmap
- Custom logging system with semantic log levels and timestamps
- Memory statistics tracking and debugging utilities
- Conversation viewer and storage inspection tools

## [0.0.3] - Factory Pattern Architecture & Modular Refactor - 2025-06-25

### Added
- pytest configuration and test infrastructure (tests to be implemented)
- Comprehensive test suite with factories, memory, and storage tests
- Factory pattern for storage with ChromaDB integration
- Modular configuration system (app, chat, debug, models configs)
- Rolling chat memory implementation
- Graceful session manager utility
- Vector store interface for extensibility

### Changed
- Refactored monolithic config.py into modular configs package
- Restructured memory system into dedicated memory module
- Reorganized storage into dedicated storage module with interface
- Moved core/temp_app.py to core/app.py for cleaner naming
- Enhanced factory pattern implementation across components
- Improved project structure with proper module organization

### Removed
- Legacy memory.py and storage.py files (replaced with modular versions)
- Old test files (test.py, test1.py) replaced with proper test structure

### Fixed
- DialoGPT response parsing for extraction
- ChromaDB collection creation error handling
- Memory stats display for simplified rolling memory system

### Tech Debt Addressed
- Eliminated monolithic config file
- Standardized logging across all components
- Improved error handling in storage initialization

## [0.0.2] - Interim Factory Pattern Refactor - 2025-06-24

### Added
- TransformerModelInterface with factory pattern for easy model switching
- Custom Logger class with semantic methods and timestamps
- Support for DialoGPT models alongside Mistral
- Modular package structure (interfaces/, factories/, core/, utils/)

### Changed
- Refactored main.py into clean entry point with `TempApp` orchestrator
- Replaced scattered print statements with consistent logging module

### Fixed
- Ctrl-C signal handling now properly saves sessions

### Breaking Changes
- Removed direct model loading - now requires Model Factory pattern

## [0.0.1] - Initial Setup - 2025-06-20

### Added
- Requirements file structure
- Main modules for auth, chat, and models
- Naive config system
- Naive pyrightconfig file
