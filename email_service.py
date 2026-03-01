"""
Email Service – high-level functions for email pipeline.
Wraps the existing email_ingestion modules.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path (if needed)
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import email pipeline modules
from email_ingestion.fetcher import GmailFetcher
from email_ingestion.pipeline import run_batch, run_build_collection_only
from email_ingestion.collection import build_collection, append_email_to_collection
from ingestion.vectorizer_e import vectorize_markdown_content, check_ollama_running

logger = logging.getLogger(__name__)

EMAILS_ROOT = Path("output/emails")
COLLECTION_DIR = EMAILS_ROOT / "collection"


def ensure_ollama() -> None:
    """Raise RuntimeError if Ollama is not running."""
    ok, _ = check_ollama_running()
    if not ok:
        raise RuntimeError("Ollama server is not running. Start it with: ollama serve")


# ----------------------------------------------------------------------
# Listing
# ----------------------------------------------------------------------
def list_emails() -> List[Dict[str, Any]]:
    """Return list of processed email folders with basic metadata."""
    if not EMAILS_ROOT.exists():
        return []
    emails = []
    for email_dir in EMAILS_ROOT.iterdir():
        if not email_dir.is_dir() or email_dir.name == "collection":
            continue
        email_id = email_dir.name
        # Look for vector mapping as indicator of processing
        mapping_path = email_dir / f"{email_id}_vector_mapping.json"
        status = "vectorized" if mapping_path.exists() else "raw"
        emails.append({
            "email_id": email_id,
            "status": status,
            "path": str(email_dir),
        })
    return emails


# ----------------------------------------------------------------------
# Fetch / Process
# ----------------------------------------------------------------------
def fetch_emails_batch(max_results: int = 50, label_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Fetch recent emails from Gmail, convert to markdown, vectorize each,
    and update the collection.
    Returns summary: processed count, skipped, errors.
    """
    ensure_ollama()
    from email_ingestion.pipeline import run_batch as original_run_batch

    # We need to capture the results – original run_batch only logs.
    # We'll just call it and then return a generic success message.
    # For a more detailed result, you could modify pipeline.py to return stats.
    try:
        original_run_batch(EMAILS_ROOT, max_results=max_results, label_ids=label_ids)
        return {"status": "success", "message": f"Batch fetch completed, check logs for details."}
    except Exception as e:
        raise RuntimeError(f"Batch fetch failed: {e}")


def rebuild_collection() -> Dict[str, Any]:
    """Rebuild the merged email collection from all processed emails."""
    ensure_ollama()
    try:
        result = build_collection(EMAILS_ROOT, COLLECTION_DIR)
        return {
            "status": "success",
            "email_count": result.get("email_count", 0),
            "chunk_count": result.get("chunk_count", 0),
            "similarity_edges": result.get("similarity_edges", 0),
        }
    except Exception as e:
        raise RuntimeError(f"Collection rebuild failed: {e}")


# ----------------------------------------------------------------------
# Email Information
# ----------------------------------------------------------------------
def get_email_info(email_id: str) -> Dict[str, Any]:
    """Return metadata for a specific email."""
    email_dir = EMAILS_ROOT / email_id
    if not email_dir.exists():
        raise ValueError(f"Email '{email_id}' not found")

    mapping_path = email_dir / f"{email_id}_vector_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return {
            "email_id": email_id,
            "status": "vectorized",
            "chunks": len(chunks),
            "path": str(email_dir),
        }
    else:
        # Possibly only markdown exists
        md_path = email_dir / f"{email_id}.md"
        if md_path.exists():
            return {
                "email_id": email_id,
                "status": "raw",
                "has_markdown": True,
                "path": str(email_dir),
            }
        else:
            return {"email_id": email_id, "status": "unknown", "path": str(email_dir)}


def get_email_markdown(email_id: str) -> str:
    """Return the markdown content of an email."""
    email_dir = EMAILS_ROOT / email_id
    if not email_dir.exists():
        raise ValueError(f"Email '{email_id}' not found")
    md_path = email_dir / f"{email_id}.md"
    if not md_path.exists():
        raise ValueError(f"No markdown file for email '{email_id}'")
    return md_path.read_text(encoding="utf-8")


def get_email_graph_stats(email_id: str) -> Dict[str, Any]:
    """Return graph statistics for a vectorized email."""
    email_dir = EMAILS_ROOT / email_id
    graph_path = email_dir / f"{email_id}_document_graph.json"
    if not graph_path.exists():
        raise ValueError(f"No document graph for email '{email_id}'")
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_types = {}
    for n in nodes:
        t = n.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    edge_relations = {}
    for e in edges:
        r = e.get("relation", "unknown")
        edge_relations[r] = edge_relations.get(r, 0) + 1
    return {
        "email_id": email_id,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": node_types,
        "edge_relations": edge_relations,
    }


# ----------------------------------------------------------------------
# Collection Query
# ----------------------------------------------------------------------
def query_email_collection(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query across all emails using the collection vector store.
    Returns answer (if you have an LLM) or just chunks. For now, we return chunks.
    """
    ensure_ollama()
    if not COLLECTION_DIR.exists():
        raise RuntimeError("Collection not built yet. Run rebuild_collection first.")

    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from ingestion.vectorizer_e import EMBEDDING_MODEL, OLLAMA_BASE_URL

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_db_path = COLLECTION_DIR / "vector_db" / "collection"
    if not vector_db_path.exists():
        raise RuntimeError("Collection vector DB not found. Run rebuild_collection.")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_path),
        collection_name="collection",
    )

    # Simple similarity search
    docs = vector_store.similarity_search(query, k=top_k)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "email_id": doc.metadata.get("email_id"),
            "chunk_id": doc.metadata.get("composite_chunk_id"),
        })

    return {
        "query": query,
        "results": results,
        "count": len(results),
    }