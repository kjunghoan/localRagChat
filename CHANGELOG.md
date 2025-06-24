# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project is meant to be a distilled module that will go into a larger project, so the
versioning is not strictly semantic but rather a reflection of the development process.

## [Unreleased]

### 0.1.0 Goals
- [X] Add a simple one way vector storage implementation
- [X] Finalize the intended structure of the codebase (as a stand-alone module)
- [ ] Simplify the files and try to keep most of the code interconnectable via dependency injection
- [ ] Write a README file

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
