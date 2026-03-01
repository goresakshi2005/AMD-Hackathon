"""
Economics / Token Usage Tracker for Document Processing Pipeline

Records token usage and estimated cost at each pipeline step (upload, vectorization,
retrieval, LLM, page summary) for stakeholder reporting and cost visibility.
All data is saved under the economics/ folder for easy export and presentation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

def _model_label(model: str) -> str:
    """Default model label from inference provider when not specified."""
    if model:
        return model
    try:
        from config.inference_config import get_provider_name
        return get_provider_name()
    except Exception:
        return "ollama"

logger = logging.getLogger(__name__)

# Folder where all economics data is stored (for stakeholder reports)
ECONOMICS_DIR = Path("economics")
# Optional: approximate cost per 1K tokens (USD) for cloud LLM comparison; Ollama local = $0
DEFAULT_INPUT_COST_PER_1K = 0.0
DEFAULT_OUTPUT_COST_PER_1K = 0.0
DEFAULT_EMBEDDING_COST_PER_1K = 0.0
# Chars per token for estimation when API doesn't return usage (~4 for English)
CHARS_PER_TOKEN_ESTIMATE = 4


def _ensure_economics_dir() -> Path:
    """Ensure economics directory exists; return path."""
    ECONOMICS_DIR.mkdir(parents=True, exist_ok=True)
    readme = ECONOMICS_DIR / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Token usage and pipeline economics for stakeholder reporting.\n"
            "Files: usage_YYYY-MM-DD.jsonl (one JSON object per line, one line per pipeline step).\n"
            "Phases: upload, pdf_processing, vectorization, retrieval, page_summary.\n"
            "Fields per line: timestamp, step, phase, document_id, input_tokens, output_tokens, "
            "embedding_tokens, total_tokens, model.\n"
            "API: GET /economics/summary?date=YYYY-MM-DD for aggregated totals by phase and step.\n",
            encoding="utf-8",
        )
    return ECONOMICS_DIR


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length (when real usage not available)."""
    if not text:
        return 0
    return max(1, len(str(text).strip()) // CHARS_PER_TOKEN_ESTIMATE)


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def log_step(
    step_name: str,
    phase: str,
    document_id: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    model: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Log a single pipeline step to the economics folder.
    phase: one of "upload" | "pdf_processing" | "vectorization" | "retrieval" | "llm" | "page_summary"
    """
    root = _ensure_economics_dir()
    record = {
        "timestamp": _timestamp(),
        "step": step_name,
        "phase": phase,
        "document_id": document_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "embedding_tokens": embedding_tokens,
        "total_tokens": input_tokens + output_tokens + embedding_tokens,
        "model": _model_label(model),
        "cost_estimate_usd": 0.0,  # Override if you add pricing
    }
    if extra:
        record["extra"] = extra

    # Append to daily JSONL file for easy aggregation
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_path = root / f"usage_{date_str}.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.debug("Economics log: %s -> %s", step_name, jsonl_path)
    return jsonl_path


def log_upload(document_id: str, file_size_bytes: int = 0, filename: str = "") -> Path:
    """Log PDF upload step (no tokens; for pipeline visibility)."""
    return log_step(
        step_name="pdf_upload",
        phase="upload",
        document_id=document_id,
        extra={"file_size_bytes": file_size_bytes, "filename": filename},
    )


def log_pdf_processing(document_id: str, total_pages: Optional[int] = None) -> Path:
    """Log PDF processing (Docling) step."""
    return log_step(
        step_name="pdf_processing",
        phase="pdf_processing",
        document_id=document_id,
        extra={"total_pages": total_pages} if total_pages is not None else None,
    )


def log_vectorization(
    document_id: str,
    embedding_tokens: int,
    llm_tokens: int,
    total_chunks: int,
    truncated_chunks: int = 0,
    model_embed: str = "nomic-embed-text",
    model_llm: str = "llama3.2:latest",
) -> Path:
    """Log vectorization step (chunk summaries + embeddings)."""
    return log_step(
        step_name="vectorization",
        phase="vectorization",
        document_id=document_id,
        input_tokens=llm_tokens,  # LLM input for summaries
        output_tokens=0,          # Could split if we had output count
        embedding_tokens=embedding_tokens,
        model=f"{model_embed}+{model_llm}",
        extra={
            "total_chunks": total_chunks,
            "truncated_chunks": truncated_chunks,
        },
    )


def log_retrieval_step(
    document_id: str,
    step_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Log a single retrieval sub-step (e.g. query_classification, analyze_chunks, generate_answer)."""
    return log_step(
        step_name=step_name,
        phase="retrieval",
        document_id=document_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
        extra=extra,
    )


def log_query_usage(
    document_id: str,
    steps: List[Dict[str, Any]],
) -> Path:
    """
    Log all token usage for one query (classification, retrieval embeddings,
    analysis, second retrieval if any, answer generation).
    steps: list of {"step": str, "input_tokens": int, "output_tokens": int, "embedding_tokens": int, "extra": dict}
    """
    root = _ensure_economics_dir()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_path = root / f"usage_{date_str}.jsonl"

    total_in = total_out = total_emb = 0
    for s in steps:
        total_in += s.get("input_tokens", 0)
        total_out += s.get("output_tokens", 0)
        total_emb += s.get("embedding_tokens", 0)
        record = {
            "timestamp": _timestamp(),
            "step": s.get("step", "unknown"),
            "phase": "retrieval",
            "document_id": document_id,
            "input_tokens": s.get("input_tokens", 0),
            "output_tokens": s.get("output_tokens", 0),
            "embedding_tokens": s.get("embedding_tokens", 0),
            "total_tokens": s.get("input_tokens", 0) + s.get("output_tokens", 0) + s.get("embedding_tokens", 0),
            "model": _model_label(""),
        }
        if s.get("extra"):
            record["extra"] = s["extra"]
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # One summary line for this query
    summary_record = {
        "timestamp": _timestamp(),
        "step": "query_total",
        "phase": "retrieval",
        "document_id": document_id,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "embedding_tokens": total_emb,
        "total_tokens": total_in + total_out + total_emb,
        "model": _model_label(""),
        "extra": {"sub_steps": len(steps)},
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    return jsonl_path


def log_page_summary(
    document_id: str,
    page_number: int,
    input_tokens_estimate: int,
    output_tokens_estimate: int,
    chunks_used: int = 0,
) -> Path:
    """Log page summarization LLM usage."""
    return log_step(
        step_name="page_summary",
        phase="page_summary",
        document_id=document_id,
        input_tokens=input_tokens_estimate,
        output_tokens=output_tokens_estimate,
        extra={"page_number": page_number, "chunks_used": chunks_used},
    )


def get_usage_summary(date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Read usage from economics/usage_<date>.jsonl and return aggregated summary.
    If date_str is None, use today (UTC).
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    root = _ensure_economics_dir()
    jsonl_path = root / f"usage_{date_str}.jsonl"
    if not jsonl_path.exists():
        return {
            "date": date_str,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_embedding_tokens": 0,
            "total_tokens": 0,
            "by_phase": {},
            "by_step": {},
            "events": 0,
        }

    total_in = total_out = total_emb = 0
    by_phase: Dict[str, Dict[str, int]] = {}
    by_step: Dict[str, Dict[str, int]] = {}
    events = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            events += 1
            ti = rec.get("input_tokens", 0)
            to = rec.get("output_tokens", 0)
            te = rec.get("embedding_tokens", 0)
            total_in += ti
            total_out += to
            total_emb += te
            phase = rec.get("phase", "unknown")
            step = rec.get("step", "unknown")
            if phase not in by_phase:
                by_phase[phase] = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
            by_phase[phase]["input_tokens"] += ti
            by_phase[phase]["output_tokens"] += to
            by_phase[phase]["embedding_tokens"] += te
            if step not in by_step:
                by_step[step] = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
            by_step[step]["input_tokens"] += ti
            by_step[step]["output_tokens"] += to
            by_step[step]["embedding_tokens"] += te

    return {
        "date": date_str,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_embedding_tokens": total_emb,
        "total_tokens": total_in + total_out + total_emb,
        "by_phase": by_phase,
        "by_step": by_step,
        "events": events,
    }
