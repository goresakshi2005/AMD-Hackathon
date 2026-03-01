"""
Inference provider abstraction: Ollama or Hugging Face.
Switch via INFERENCE_PROVIDER=ollama|huggingface (default: ollama).
Exposes get_embeddings(), get_llm(), check_inference_ready().
"""

# Load .env from project root so INFERENCE_PROVIDER and token are available
try:
    from pathlib import Path
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def _normalize_embedding_result(raw: Any, num_texts: int) -> List[List[float]]:
    """Ensure embed_documents return value is List[List[float]] for Chroma compatibility."""
    if raw is None:
        return [[]] * num_texts
    if isinstance(raw, dict):
        # Some APIs return {"embeddings": [[...], ...]} or similar
        emb = raw.get("embeddings") or raw.get("embedding")
        if emb is not None:
            raw = emb
        else:
            # Fallback: use first value that looks like a list of numbers
            for v in raw.values():
                if isinstance(v, list) and (not v or isinstance(v[0], (int, float))):
                    raw = v if (v and isinstance(v[0], list)) else [v]
                    break
    if isinstance(raw, list) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, (int, float)):
            # Single embedding returned as one list of floats
            return [raw]
        if isinstance(first, list):
            return raw
    return raw if isinstance(raw, list) else [raw]


class _NormalizedEmbeddings:
    """Wraps an embedding client so embed_documents always returns List[List[float]]."""

    def __init__(self, client: Any):
        self._client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw = self._client.embed_documents(texts)
        return _normalize_embedding_result(raw, len(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class _HFInferenceClientEmbeddings:
    """LangChain Embeddings using huggingface_hub InferenceClient (Inference Providers)."""

    def __init__(self, token: str, model: str):
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(token=token)
        self._model = model

    def _to_list(self, arr) -> List[float]:
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        if isinstance(arr, list) and arr and isinstance(arr[0], list):
            return arr[0]
        return arr if isinstance(arr, list) else list(arr)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out = []
        try:
            for t in texts:
                emb = self._client.feature_extraction(t, model=self._model)
                out.append(self._to_list(emb))
        except StopIteration:
            raise ValueError(
                f"No Inference Provider supports feature-extraction for model '{self._model}'. "
                "Set HF_EMBEDDING_MODEL to a supported model (e.g. sentence-transformers/all-mpnet-base-v2, "
                "thenlper/gte-large) or use Ollama for embeddings. See https://huggingface.co/inference/models"
            ) from None
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------- CONFIGURATION ----------------
INFERENCE_PROVIDER = os.environ.get("INFERENCE_PROVIDER", "ollama").strip().lower()
if INFERENCE_PROVIDER not in ("ollama", "huggingface"):
    INFERENCE_PROVIDER = "ollama"
    logger.warning("INFERENCE_PROVIDER must be 'ollama' or 'huggingface'; using 'ollama'")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b")

# Hugging Face (Option A: serverless Inference API)
HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
HF_LLM_MODEL = os.environ.get("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", os.environ.get("HF_TOKEN", ""))


def get_embeddings():
    """Return a LangChain-compatible embedding instance (Ollama or Hugging Face)."""
    if INFERENCE_PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    else:
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) is required when INFERENCE_PROVIDER=huggingface. "
                "Set it in the environment or .env file."
            )
        # Use InferenceClient (Inference Providers); router is chat-only and api-inference is deprecated.
        try:
            return _HFInferenceClientEmbeddings(
                token=HUGGINGFACEHUB_API_TOKEN,
                model=HF_EMBEDDING_MODEL,
            )
        except Exception as e:
            logger.warning("Hugging Face InferenceClient embeddings failed: %s", e)
            try:
                from langchain_huggingface import HuggingFaceEndpointEmbeddings
                return _NormalizedEmbeddings(HuggingFaceEndpointEmbeddings(
                    repo_id=HF_EMBEDDING_MODEL,
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                ))
            except ImportError:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                return _NormalizedEmbeddings(HuggingFaceInferenceAPIEmbeddings(
                    model_name=HF_EMBEDDING_MODEL,
                    api_key=HUGGINGFACEHUB_API_TOKEN,
                ))


def get_llm(temperature: float = 0.3, **kwargs) -> Any:
    """Return a LangChain-compatible LLM instance (Ollama or Hugging Face)."""
    if INFERENCE_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            **kwargs,
        )
    else:
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) is required when INFERENCE_PROVIDER=huggingface. "
                "Set it in the environment or .env file."
            )
        # Hugging Face now uses router.huggingface.co (OpenAI-compatible); api-inference.huggingface.co returns 410 Gone.
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HUGGINGFACEHUB_API_TOKEN,
                model=HF_LLM_MODEL,
                temperature=temperature,
                **kwargs,
            )
        except ImportError:
            base_url = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"
            try:
                from langchain_huggingface import HuggingFaceEndpoint
                return HuggingFaceEndpoint(
                    endpoint_url=base_url,
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                    temperature=temperature,
                    **kwargs,
                )
            except ImportError:
                from langchain_community.llms import HuggingFaceEndpoint
                return HuggingFaceEndpoint(
                    endpoint_url=base_url,
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                    task="text-generation",
                    temperature=temperature,
                    model_kwargs=dict(kwargs) if kwargs else {},
                )


def check_inference_ready() -> Tuple[bool, List[str]]:
    """
    Check if the configured inference backend is ready.
    Returns (is_ready, list_of_model_names_or_info).
    """
    if INFERENCE_PROVIDER == "ollama":
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
            response = urllib.request.urlopen(req, timeout=5)
            data = json.loads(response.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            return True, models
        except urllib.error.URLError:
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                response = urllib.request.urlopen(req, timeout=5)
                data = json.loads(response.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return True, models
            except Exception:
                return False, []
        except Exception as e:
            logger.debug("Ollama check error: %s", e)
            return False, []
    else:
        if not HUGGINGFACEHUB_API_TOKEN:
            return False, []
        try:
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"},
            )
            response = urllib.request.urlopen(req, timeout=10)
            data = json.loads(response.read().decode())
            name = data.get("name", "unknown")
            return True, [f"Hugging Face ({name})"]
        except Exception as e:
            logger.debug("Hugging Face check error: %s", e)
            return False, []


def get_provider_name() -> str:
    """Return current provider name for logging/API (e.g. 'ollama', 'huggingface')."""
    return INFERENCE_PROVIDER


def get_embedding_model_id() -> str:
    """Return the active embedding model id (for economics/logging)."""
    if INFERENCE_PROVIDER == "ollama":
        return OLLAMA_EMBEDDING_MODEL
    return HF_EMBEDDING_MODEL


def get_llm_model_id() -> str:
    """Return the active LLM model id (for economics/logging)."""
    if INFERENCE_PROVIDER == "ollama":
        return OLLAMA_LLM_MODEL
    return HF_LLM_MODEL
