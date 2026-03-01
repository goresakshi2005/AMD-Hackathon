"""
Configuration package. Exposes inference backend (Ollama / Hugging Face).
"""
from config.inference_config import (
    check_inference_ready,
    get_embeddings,
    get_embedding_model_id,
    get_llm,
    get_llm_model_id,
    get_provider_name,
)

__all__ = [
    "check_inference_ready",
    "get_embeddings",
    "get_embedding_model_id",
    "get_llm",
    "get_llm_model_id",
    "get_provider_name",
]
