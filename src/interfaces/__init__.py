"""
Abstract interfaces for the Local RAG Chat system.

These interfaces define the contracts that different implementations must follow,
enabling easy swapping of models, memory systems, and storage backends.
"""

from .model import TransformerModelInterface, ModelConfig

__all__ = [
    "TransformerModelInterface", 
    "ModelConfig"
]
