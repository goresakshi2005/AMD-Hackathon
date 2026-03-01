"""
Shared document pipeline service layer.
Used by both the FastAPI app (main.py) and the MCP server (mcp_server.py).
Raises normal Python exceptions (ValueError, FileNotFoundError) so callers can
map them to HTTP responses or MCP tool error messages.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import networkx as nx

from config.inference_config import get_llm
from detection import process_single_pdf
from economics_tracker import (
    log_pdf_processing,
    log_query_usage,
    log_upload,
    log_vectorization,
    log_page_summary,
)
from page_summarization import load_page_agent
from retrivalAgentE import (
    create_retrieval_agent,
    set_agent_resources,
    load_chunks_from_mapping,
    load_vector_store,
    find_vector_mapping_file,
    find_graph_file,
    find_vector_db_path,
    DocumentGraph as RetrievalDocumentGraph,
    AgentState,
)
from vectorizerE import (
    create_vectorization_workflow,
    VectorizerState,
    DocumentGraph,
)

logger = logging.getLogger(__name__)

# Configuration - must match main.py layout
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")

_loaded_agents: Dict[str, Dict[str, Any]] = {}


def get_document_path(document_id: str) -> Path:
    """Get document output directory."""
    return OUTPUT_DIR / document_id


def get_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document information as a dict (same keys as main.DocumentInfo).
    Returns None if the document does not exist.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        return None

    md_files = list(doc_path.glob("*.md"))
    md_path = md_files[0] if md_files else None

    page_mapping_path = None
    if md_path:
        page_mapping_path = doc_path / f"{md_path.stem}_page_mapping.json"
        if not page_mapping_path.exists():
            page_mapping_path = None

    confidence_path = None
    layout_score = ocr_score = parse_score = table_score = None
    mean_grade = low_grade = None
    if md_path:
        potential_conf_path = doc_path / f"{md_path.stem}_confidence.json"
        if potential_conf_path.exists():
            confidence_path = potential_conf_path
            try:
                with open(confidence_path, "r", encoding="utf-8") as f:
                    confidence_data = json.load(f)
                layout_score = confidence_data.get("layout_score")
                ocr_score = confidence_data.get("ocr_score")
                parse_score = confidence_data.get("parse_score")
                table_score = confidence_data.get("table_score")
                mean_grade = confidence_data.get("mean_grade")
                low_grade = confidence_data.get("low_grade")
            except Exception:
                confidence_path = None

    plan_e_dir = doc_path / "E"
    vector_mapping_path = None
    graph_path = None
    vector_db_path = None
    total_chunks = None

    if plan_e_dir.exists():
        mapping_file = find_vector_mapping_file(plan_e_dir)
        if mapping_file:
            vector_mapping_path = str(mapping_file)
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    total_chunks = len(json.load(f))
            except Exception:
                pass
        graph_file = find_graph_file(plan_e_dir)
        if graph_file:
            graph_path = str(graph_file)
        vector_db = find_vector_db_path(plan_e_dir)
        if vector_db:
            vector_db_path = str(vector_db)

    status = "uploaded"
    if md_path:
        status = "processing"
    if vector_mapping_path and graph_path:
        status = "vectorized"
    if vector_mapping_path and graph_path and vector_db_path:
        status = "ready"

    total_pages = None
    if page_mapping_path:
        try:
            with open(page_mapping_path, "r", encoding="utf-8") as f:
                total_pages = json.load(f).get("total_pages")
        except Exception:
            pass

    return {
        "document_id": document_id,
        "name": doc_path.name,
        "status": status,
        "markdown_path": str(md_path) if md_path else None,
        "page_mapping_path": str(page_mapping_path) if page_mapping_path else None,
        "vector_mapping_path": vector_mapping_path,
        "graph_path": graph_path,
        "vector_db_path": vector_db_path,
        "confidence_path": str(confidence_path) if confidence_path else None,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "layout_score": layout_score,
        "ocr_score": ocr_score,
        "parse_score": parse_score,
        "table_score": table_score,
        "mean_grade": mean_grade,
        "low_grade": low_grade,
    }


def load_agent_for_document(document_id: str) -> Dict[str, Any]:
    """Load agent resources for a document. Raises ValueError if not vectorized or files missing."""
    if document_id in _loaded_agents:
        return _loaded_agents[document_id]

    doc_path = get_document_path(document_id)
    plan_e_dir = doc_path / "E"

    if not plan_e_dir.exists():
        raise ValueError(f"Document {document_id} not vectorized. Run vectorization first.")

    vector_mapping_file = find_vector_mapping_file(plan_e_dir)
    graph_file = find_graph_file(plan_e_dir)
    vdb_path = find_vector_db_path(plan_e_dir)

    if not vector_mapping_file or not vector_mapping_file.exists():
        raise ValueError("Vector mapping file not found")
    if not graph_file or not graph_file.exists():
        raise ValueError("Graph file not found")
    if not vdb_path or not vdb_path.exists():
        raise ValueError("Vector database not found")

    chunks = load_chunks_from_mapping(vector_mapping_file)
    document_graph = RetrievalDocumentGraph()
    document_graph.load(graph_file)
    vector_store = load_vector_store(vdb_path)
    llm = get_llm(temperature=0.3)
    agent = create_retrieval_agent(vector_store, document_graph, chunks, llm, doc_path)
    set_agent_resources(vector_store, document_graph, chunks, llm, doc_path)

    _loaded_agents[document_id] = {
        "agent": agent,
        "vector_store": vector_store,
        "document_graph": document_graph,
        "chunks": chunks,
        "llm": llm,
    }
    return _loaded_agents[document_id]


def run_detection_for_document(document_id: str) -> None:
    """
    Run PDF detection (steps 3–6) for a document whose PDF is already in the document folder.
    Used by main.py after saving an uploaded file. Raises ValueError if no PDF found in folder.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")
    pdfs = list(doc_path.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF file found in document folder {doc_path}")
    pdf_path = pdfs[0]
    process_single_pdf(pdf_path, OUTPUT_DIR, target_dir=doc_path)
    total_pages = None
    md_files = list(doc_path.glob("*.md"))
    if md_files:
        pm_path = doc_path / f"{md_files[0].stem}_page_mapping.json"
        if pm_path.exists():
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    total_pages = json.load(f).get("total_pages")
            except Exception:
                pass
    log_pdf_processing(document_id, total_pages=total_pages)


def upload_pdf_from_path(file_path: Path) -> str:
    """
    Copy PDF to a new document folder and run detection (steps 3–6). Returns document_id.
    Raises FileNotFoundError if file_path does not exist; ValueError if not a PDF.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    document_id = str(uuid4())
    doc_path = get_document_path(document_id)
    doc_path.mkdir(parents=True, exist_ok=True)
    dest_pdf = doc_path / file_path.name
    shutil.copy2(file_path, dest_pdf)
    log_upload(document_id, file_size_bytes=dest_pdf.stat().st_size, filename=file_path.name)

    process_single_pdf(dest_pdf, OUTPUT_DIR, target_dir=doc_path)

    total_pages = None
    md_files = list(doc_path.glob("*.md"))
    if md_files:
        pm_path = doc_path / f"{md_files[0].stem}_page_mapping.json"
        if pm_path.exists():
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    total_pages = json.load(f).get("total_pages")
            except Exception:
                pass
    log_pdf_processing(document_id, total_pages=total_pages)
    return document_id


def trigger_vectorize(document_id: str) -> None:
    """
    Run vectorization for the document (synchronous). Raises ValueError if document
    not found or markdown missing.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")
    md_files = list(doc_path.glob("*.md"))
    if not md_files:
        raise ValueError("Document must be processed first (markdown not found)")

    logger.info("Starting vectorization for document: %s", document_id)
    initial_state: VectorizerState = {
        "markdown_file": str(doc_path),
        "chunks": [],
        "structure": {},
        "processed_chunks": [],
        "vector_store": None,
        "document_graph": DocumentGraph(),
        "json_mapping": [],
        "page_mapping": None,
        "page_classifications": None,
        "output_folder": str(doc_path),
        "token_usage": None,
    }
    workflow = create_vectorization_workflow()
    final_state = workflow.invoke(initial_state)

    usage = final_state.get("token_usage")
    if usage:
        log_vectorization(
            document_id,
            embedding_tokens=usage.get("embedding_tokens", 0),
            llm_tokens=usage.get("llm_tokens", 0),
            total_chunks=usage.get("total_chunks", 0),
            truncated_chunks=usage.get("truncated_chunks", 0),
        )
    logger.info("Vectorization complete for document: %s", document_id)


def query_document(
    document_id: str,
    query: str,
    include_chunks: bool = True,
) -> Dict[str, Any]:
    """
    Run RAG query over a vectorized document. Returns dict with answer, retrieval_stats,
    chunks (if include_chunks), chunk_analysis, debug_info. Raises ValueError if document
    not ready or query empty.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty")

    agent_resources = load_agent_for_document(document_id)
    agent = agent_resources["agent"]

    initial_state: AgentState = {
        "query": query,
        "is_page_summary": False,
        "page_number": None,
        "seed_chunk_ids": [],
        "seed_chunk_scores": {},
        "graph_expanded_ids": [],
        "retrieved_chunks": [],
        "reranked_chunks": None,
        "rerank_scores": {},
        "chunk_analysis": "",
        "needs_more_info": False,
        "new_query": None,
        "second_seed_ids": [],
        "second_seed_scores": {},
        "second_expanded_ids": [],
        "second_retrieval_chunks": [],
        "final_answer": "",
        "iteration_count": 0,
        "document_folder": None,
        "debug_info": {},
        "token_usage": [],
    }

    final_state = agent.invoke(initial_state)
    if final_state.get("token_usage"):
        log_query_usage(document_id, final_state["token_usage"])

    seed_chunk_ids = set(final_state.get("seed_chunk_ids", []))
    graph_expanded_ids = set(final_state.get("graph_expanded_ids", []))
    second_seed_ids = set(final_state.get("second_seed_ids", []))
    second_expanded_ids = set(final_state.get("second_expanded_ids", []))
    seed_chunk_scores = final_state.get("seed_chunk_scores", {})
    second_seed_scores = final_state.get("second_seed_scores", {})
    rerank_scores = final_state.get("rerank_scores", {})
    reranked_chunks = final_state.get("reranked_chunks")
    total_initial = len(reranked_chunks) if reranked_chunks else len(final_state.get("retrieved_chunks", []))

    retrieval_stats = {
        "seed_chunks": len(seed_chunk_ids),
        "graph_expanded_chunks": len(graph_expanded_ids),
        "total_initial_chunks": total_initial,
        "second_seed_chunks": len(second_seed_ids),
        "second_expanded_chunks": len(second_expanded_ids),
        "total_second_chunks": len(final_state.get("second_retrieval_chunks", [])),
        "total_chunks_used": total_initial + len(final_state.get("second_retrieval_chunks", [])),
        "iterations": final_state.get("iteration_count", 0),
        "reranking_applied": reranked_chunks is not None,
        "reranked_chunks_count": len(reranked_chunks) if reranked_chunks else 0,
    }
    if final_state.get("new_query"):
        retrieval_stats["second_query"] = final_state["new_query"]

    chunks_out: List[Dict[str, Any]] = []
    if include_chunks:
        for chunk in final_state.get("retrieved_chunks", []):
            cid = chunk.metadata.get("chunk_index")
            if cid is None:
                continue
            source = "seed" if cid in seed_chunk_ids else ("graph_expanded" if cid in graph_expanded_ids else "initial")
            chunks_out.append({
                "chunk_index": cid,
                "content": chunk.page_content,
                "heading": chunk.metadata.get("heading", "No heading"),
                "section_path": chunk.metadata.get("section_path", ""),
                "section_title": chunk.metadata.get("section_title", ""),
                "page_number": chunk.metadata.get("page_number"),
                "page_classification": chunk.metadata.get("page_classification"),
                "summary": chunk.metadata.get("summary", ""),
                "chunk_type": chunk.metadata.get("chunk_type", "text"),
                "has_table": chunk.metadata.get("has_table", False),
                "table_context": chunk.metadata.get("table_context"),
                "start_line": chunk.metadata.get("start_line"),
                "content_length": len(chunk.page_content),
                "retrieval_source": source,
                "similarity_score": seed_chunk_scores.get(cid),
                "rerank_score": rerank_scores.get(cid),
            })
        for chunk in final_state.get("second_retrieval_chunks", []):
            cid = chunk.metadata.get("chunk_index")
            if cid is None or any(c["chunk_index"] == cid for c in chunks_out):
                continue
            source = "second_seed" if cid in second_seed_ids else ("second_expanded" if cid in second_expanded_ids else "second_retrieval")
            chunks_out.append({
                "chunk_index": cid,
                "content": chunk.page_content,
                "heading": chunk.metadata.get("heading", "No heading"),
                "section_path": chunk.metadata.get("section_path", ""),
                "section_title": chunk.metadata.get("section_title", ""),
                "page_number": chunk.metadata.get("page_number"),
                "page_classification": chunk.metadata.get("page_classification"),
                "summary": chunk.metadata.get("summary", ""),
                "chunk_type": chunk.metadata.get("chunk_type", "text"),
                "has_table": chunk.metadata.get("has_table", False),
                "table_context": chunk.metadata.get("table_context"),
                "start_line": chunk.metadata.get("start_line"),
                "content_length": len(chunk.page_content),
                "retrieval_source": source,
                "similarity_score": second_seed_scores.get(cid),
                "rerank_score": rerank_scores.get(cid),
            })
        chunks_out.sort(key=lambda c: c["chunk_index"])

    debug_info = dict(final_state.get("debug_info", {}))
    if rerank_scores:
        debug_info["rerank_scores_summary"] = {
            "total_reranked": len(rerank_scores),
            "avg_score": sum(rerank_scores.values()) / len(rerank_scores),
            "max_score": max(rerank_scores.values()),
            "min_score": min(rerank_scores.values()),
        }

    return {
        "answer": final_state.get("final_answer", "No answer generated"),
        "document_id": document_id,
        "query": query,
        "retrieval_stats": retrieval_stats,
        "chunk_analysis": final_state.get("chunk_analysis"),
        "chunks": chunks_out,
        "debug_info": debug_info,
    }


def summarize_page(document_id: str, page_number: int) -> Dict[str, Any]:
    """
    Generate page-level summary. Returns dict with page_number, summary, key_points,
    sections, has_content, used_adjacent_pages, adjacent_pages_used, page_classification,
    chunks_used, total_chunks. Raises ValueError if document not found or not vectorized.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")
    plan_e_dir = doc_path / "E"
    if not plan_e_dir.exists():
        raise ValueError("Document not vectorized. Run vectorization first.")

    agent = load_page_agent(doc_path)
    if not agent:
        raise ValueError("Failed to load page summarization agent")

    summary_result = agent.summarize_page(page_number, use_adjacent_if_empty=True)
    chunks_used = len(summary_result.chunks_used) if summary_result.chunks_used else 0
    input_est = (chunks_used * 500 + 500) // 4
    output_est = max(1, len(summary_result.summary) // 4)
    log_page_summary(document_id, page_number, input_tokens_estimate=input_est, output_tokens_estimate=output_est, chunks_used=chunks_used)

    return {
        "page_number": summary_result.page_number,
        "summary": summary_result.summary,
        "key_points": summary_result.key_points,
        "sections": summary_result.sections,
        "has_content": summary_result.has_content,
        "used_adjacent_pages": summary_result.used_adjacent_pages,
        "adjacent_pages_used": summary_result.adjacent_pages_used,
        "page_classification": summary_result.page_classification,
        "chunks_used": summary_result.chunks_used or [],
        "total_chunks": chunks_used,
    }


def get_graph_stats(document_id: str) -> Dict[str, Any]:
    """
    Return graph node/edge counts and optional similarity stats. Returns empty-style
    stats if document or graph not found (no exception).
    """
    from visualizeGraphE import load_graph, find_graph_file as find_viz_graph_file

    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")

    plan_e_dir = doc_path / "E"
    empty = {
        "total_nodes": 0,
        "total_edges": 0,
        "node_types": {},
        "edge_relations": {},
        "density": 0.0,
    }
    if not plan_e_dir.exists():
        return empty

    graph_file = find_viz_graph_file(plan_e_dir)
    if not graph_file or not graph_file.exists():
        return empty

    try:
        G = load_graph(graph_file)
        node_types: Dict[str, int] = {}
        for nid in G.nodes():
            t = G.nodes[nid].get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
        edge_relations: Dict[str, int] = {}
        for _u, _v, d in G.edges(data=True):
            r = d.get("relation", "unknown")
            edge_relations[r] = edge_relations.get(r, 0) + 1
        stats = {
            "total_nodes": len(G.nodes),
            "total_edges": len(G.edges),
            "node_types": node_types,
            "edge_relations": edge_relations,
            "density": float(nx.density(G)) if len(G.nodes) > 0 else 0.0,
        }
        if "similar_to" in edge_relations:
            sim_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == "similar_to"]
            if sim_edges:
                sims = [d.get("similarity", 0.0) for u, v, d in sim_edges if d.get("similarity")]
                if sims:
                    stats["similarity_stats"] = {
                        "count": len(sims),
                        "average": sum(sims) / len(sims),
                        "min": min(sims),
                        "max": max(sims),
                    }
        return stats
    except Exception as e:
        logger.error("Error getting graph stats: %s", e, exc_info=True)
        raise ValueError(f"Failed to get stats: {e}") from e


def get_document_markdown(document_id: str) -> str:
    """Return full markdown content for the document. Raises ValueError if not found."""
    info = get_document_info(document_id)
    if not info or not info.get("markdown_path"):
        raise ValueError("Markdown file not found for document")
    md_path = Path(info["markdown_path"])
    if not md_path.exists():
        raise ValueError("Markdown file not found")
    return md_path.read_text(encoding="utf-8")
