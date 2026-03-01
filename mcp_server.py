"""
MCP (Model Context Protocol) server for the Document Processing Pipeline.
Exposes pipeline operations as tools for AI clients (e.g. Cursor, Claude Desktop).

Usage:
  stdio (default, for Cursor):  python mcp_server.py
  Streamable HTTP (for MCP Inspector):  python mcp_server.py --http
  Then in MCP Inspector use URL: http://127.0.0.1:8765/mcp
"""

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Force UTF-8 on stderr so emoji prints don't crash on Windows (cp1252)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Prevent warnings from polluting the MCP stdio JSON-RPC stream
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*was deprecated.*")

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Silence stdout/stderr during imports to avoid non-JSON lines emitted at module
# import time (these would break the JSON-RPC handshake with the client).
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        from services import document_service
        from web_researcher import AdvancedResearcher
        from meet_knowledgeGraph import MeetingProcessor
        from meet_taskScheduler import MeetingTaskParser, CalendarScheduler
        # Email service (optional)
        try:
            import email_service
            EMAIL_SERVICE_AVAILABLE = True
        except ImportError:
            email_service = None
            EMAIL_SERVICE_AVAILABLE = False

MEETINGS_OUTPUT_ROOT = Path("output/meetings")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("MCP SDK not installed. Run: pip install -r requirements-mcp.txt", file=sys.stderr)
    sys.exit(1)

MCP_HTTP_PORT = int(os.environ.get("MCP_HTTP_PORT", "8765"))

mcp = FastMCP(
    "Document Processing Pipeline",
    json_response=True,
    host="127.0.0.1",
    port=MCP_HTTP_PORT,
)

# Shared researcher instance (reuses disk cache across calls)
_researcher = AdvancedResearcher()


# ============================================================================
# Internal helpers
# ============================================================================

def _silent(fn, *args, **kwargs):
    """Run fn(*args) suppressing all stdout/stderr so prints never touch the MCP stream."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)


def _parse_with_openai(query: str, results) -> dict:
    """
    Send raw search results to OpenAI and return a structured dict:
        summary       – 2-3 sentence executive overview
        key_findings  – list of 4-6 finding strings
        sections      – list of {heading, content} deep-dive blocks
        conclusion    – closing paragraph
        raw_markdown  – full Markdown answer with inline citations
    Falls back to plain researcher synthesis when OpenAI is unavailable.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # ── Fallback: no OpenAI key ──────────────────────────────────────────────
    if not OPENAI_API_KEY:
        raw = _silent(_researcher.synthesize_results, query, results)
        return {
            "summary": "",
            "key_findings": [],
            "sections": [],
            "conclusion": "",
            "raw_markdown": raw,
        }

    # ── Build compact source text for the prompt ─────────────────────────────
    sources_text = ""
    for i, r in enumerate(results[:10], 1):
        sources_text += (
            f"\n[{i}] {r.source} | {r.title}\n"
            f"URL: {r.url}\n"
            f"Content: {r.content[:600]}\n"
        )

    system_prompt = (
        "You are an expert research analyst. "
        "Given a set of web search results, produce a structured JSON response. "
        "Return ONLY valid JSON — no markdown fences, no extra commentary. "
        "The JSON must contain exactly these keys:\n"
        "  summary       : string  – 2-3 sentence executive overview\n"
        "  key_findings  : array of strings – 4 to 6 concise bullet points\n"
        "  sections      : array of objects {\"heading\": string, \"content\": string} "
        "– 2 to 4 deep-dive sections expanding on the findings\n"
        "  conclusion    : string  – a short closing paragraph\n"
        "  raw_markdown  : string  – the complete answer as clean Markdown "
        "with inline citations like [1], [2] matching the source numbers provided\n"
    )

    user_prompt = (
        f"Research query: {query}\n\n"
        f"Search results:\n{sources_text}"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
        )

        raw_text = response.choices[0].message.content.strip()

        # Strip accidental markdown code fences
        if raw_text.startswith("```"):
            parts = raw_text.split("```")
            raw_text = parts[1]
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text.strip())

        # Guarantee all keys exist
        parsed.setdefault("summary",      "")
        parsed.setdefault("key_findings", [])
        parsed.setdefault("sections",     [])
        parsed.setdefault("conclusion",   "")
        parsed.setdefault("raw_markdown", "")
        return parsed

    except Exception as e:
        # Graceful fallback
        raw = _silent(_researcher.synthesize_results, query, results)
        return {
            "summary":      "",
            "key_findings": [],
            "sections":     [],
            "conclusion":   "",
            "raw_markdown": raw,
            "parse_error":  str(e)[:200],
        }


# ============================================================================
# Document Tools
# ============================================================================

@mcp.tool()
def list_documents() -> str:
    """List all processed documents. Returns a JSON list with document_id, status, total_pages, total_chunks for each."""
    documents = []
    if document_service.OUTPUT_DIR.exists():
        for doc_dir in document_service.OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                doc_info = document_service.get_document_info(doc_dir.name)
                if doc_info:
                    documents.append({
                        "document_id":  doc_info["document_id"],
                        "name":         doc_info["name"],
                        "status":       doc_info["status"],
                        "total_pages":  doc_info.get("total_pages"),
                        "total_chunks": doc_info.get("total_chunks"),
                    })
    return json.dumps(documents, indent=2)


@mcp.tool()
def get_document_info(document_id: str) -> str:
    """Get metadata for one document (status, paths, stats). Returns JSON or an error message if not found."""
    doc_info = document_service.get_document_info(document_id)
    if not doc_info:
        return json.dumps({"error": f"Document '{document_id}' not found"})
    return json.dumps(doc_info, indent=2)


@mcp.tool()
def upload_document(file_path: str) -> str:
    """Upload a PDF from a local file path. Copies the file, runs detection (steps 3-6), and returns the document_id."""
    try:
        doc_id = document_service.upload_pdf_from_path(Path(file_path))
        return json.dumps({"document_id": doc_id, "status": "processing",
                           "message": "PDF uploaded and processing complete"})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Upload failed: {e}"})


@mcp.tool()
def vectorize_document(document_id: str) -> str:
    """Run vectorization for a document. Synchronous; may take several minutes for large documents."""
    try:
        document_service.trigger_vectorize(document_id)
        return json.dumps({"document_id": document_id, "status": "ok",
                           "message": "Vectorization complete"})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Vectorization failed: {e}"})


@mcp.tool()
def query_document(document_id: str, query: str, include_chunks: bool = True) -> str:
    """RAG query over a vectorized document. Returns answer, retrieval_stats, and optional chunks."""
    try:
        result = document_service.query_document(document_id, query, include_chunks=include_chunks)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Query failed: {e}"})


@mcp.tool()
def summarize_page(document_id: str, page_number: int) -> str:
    """Generate a page-level summary for a given page number (1-based)."""
    try:
        result = document_service.summarize_page(document_id, page_number)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Summarize failed: {e}"})


@mcp.tool()
def get_graph_stats(document_id: str) -> str:
    """Get graph node/edge counts and optional similarity stats for a document."""
    try:
        stats = document_service.get_graph_stats(document_id)
        return json.dumps(stats, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Failed to get stats: {e}"})


@mcp.tool()
def get_document_markdown(document_id: str) -> str:
    """Return the full markdown content of the document (useful for the AI to read full text)."""
    try:
        content = document_service.get_document_markdown(document_id)
        return content
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Failed to get markdown: {e}"})


# ============================================================================
# Web Research Tool
# ============================================================================

@mcp.tool()
def web_research(
    query: str,
    sources: str = "all",
    max_results_per_source: int = 7,
) -> str:
    """
    Research any topic using multiple search engines, then parse and structure
    the results with OpenAI into summary, key findings, sections and conclusion.

    Args:
        query: The research question or topic to investigate.
        sources: Comma-separated source list, or "all".
                 Options: tavily, exa, google, duckduckgo, wikipedia.
        max_results_per_source: Results per source, 1-10 (default 7).

    Returns JSON with:
        query, summary, key_findings, sections, conclusion,
        raw_markdown, num_sources, sources, timestamp, errors
    """
    valid_sources = {"tavily", "exa", "google", "duckduckgo", "wikipedia"}

    # Resolve active sources
    if sources.strip().lower() == "all":
        active_sources = list(valid_sources)
    else:
        active_sources = [
            s.strip().lower()
            for s in sources.split(",")
            if s.strip().lower() in valid_sources
        ]
        if not active_sources:
            return json.dumps({
                "error": (
                    "No valid sources specified. "
                    f"Choose from: {', '.join(sorted(valid_sources))}"
                )
            })

    n = min(max(1, max_results_per_source), 10)

    source_method_map = {
        "tavily":     _researcher.tavily_search,
        "exa":        _researcher.exa_search,
        "google":     _researcher.serper_search,
        "duckduckgo": _researcher.duckduckgo_search,
        "wikipedia":  _researcher.wikipedia_search,
    }

    # ── 1. Gather raw results ────────────────────────────────────────────────
    all_results = []
    errors: dict = {}

    for src in active_sources:
        try:
            results = _silent(source_method_map[src], query)
            all_results.extend(results[:n])
        except Exception as e:
            errors[src] = str(e)[:120]

    # ── 2. Deduplicate ───────────────────────────────────────────────────────
    unique = _silent(_researcher.deduplicate_results, all_results)

    # ── 3. Parse & structure with OpenAI ────────────────────────────────────
    parsed = _parse_with_openai(query, unique)

    if "parse_error" in parsed:
        errors["openai_parse"] = parsed.pop("parse_error")

    return json.dumps({
        "query":        query,
        "summary":      parsed.get("summary",      ""),
        "key_findings": parsed.get("key_findings", []),
        "sections":     parsed.get("sections",     []),
        "conclusion":   parsed.get("conclusion",   ""),
        "raw_markdown": parsed.get("raw_markdown", ""),
        "num_sources":  len(unique),
        "sources":      [[r.title, r.url, r.source] for r in unique[:10]],
        "timestamp":    datetime.now().isoformat(),
        "errors":       errors,
    }, indent=2)


# ============================================================================
# Meeting Tools
# ============================================================================

@mcp.tool()
def list_meetings() -> str:
    """
    List all processed meetings stored in output/meetings.
    Returns JSON list with meeting_name, tasks_count, has_graph, output_dir for each.
    """
    meetings = []
    if MEETINGS_OUTPUT_ROOT.exists():
        for meeting_dir in sorted(MEETINGS_OUTPUT_ROOT.iterdir()):
            if not meeting_dir.is_dir():
                continue
            name = meeting_dir.name
            tasks_file = meeting_dir / f"{name}_tasks.json"
            kg_file    = meeting_dir / f"{name}_knowledge_graph.json"
            tasks_count = 0
            if tasks_file.exists():
                try:
                    data = json.loads(tasks_file.read_text(encoding="utf-8"))
                    tasks_count = len(data) if isinstance(data, list) else 1
                except Exception:
                    pass
            meetings.append({
                "meeting_name": name,
                "tasks_count":  tasks_count,
                "has_graph":    kg_file.exists(),
                "has_tasks":    tasks_file.exists(),
                "output_dir":   str(meeting_dir),
            })
    return json.dumps(meetings, indent=2)


@mcp.tool()
def get_meeting_tasks(meeting_name: str) -> str:
    """
    Return the extracted tasks for a processed meeting (JSON array).
    Use list_meetings() first to find valid meeting_name values.
    """
    tasks_file = MEETINGS_OUTPUT_ROOT / meeting_name / f"{meeting_name}_tasks.json"
    if not tasks_file.exists():
        # Fallback: extract from knowledge graph
        kg_file = MEETINGS_OUTPUT_ROOT / meeting_name / f"{meeting_name}_knowledge_graph.json"
        if kg_file.exists():
            try:
                data = json.loads(kg_file.read_text(encoding="utf-8"))
                tasks = [
                    n for n in data.get("nodes", []) if n.get("type") == "task"
                ]
                return json.dumps(tasks, indent=2)
            except Exception as e:
                return json.dumps({"error": f"Failed to read graph: {e}"})
        return json.dumps({"error": f"Meeting '{meeting_name}' not found or not yet processed"})
    try:
        return tasks_file.read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def process_meeting(audio_file_path: str) -> str:
    """
    Transcribe an audio/video file, extract tasks, entities, and build a
    knowledge graph. Saves results to output/meetings/<name>/.

    Args:
        audio_file_path: Absolute or relative path to the audio file
                         (mp3, mp4, wav, m4a, ogg, flac, etc.).

    Returns JSON with meeting_name, transcript_length, tasks count, graph stats,
    and output_dir.
    """
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        return json.dumps({"error": f"File not found: {audio_file_path}"})

    try:
        processor = MeetingProcessor(audio_path)
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                summary = processor.process()
        if summary is None:
            return json.dumps({"error": "Processing failed — check that the audio file is valid."})
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Meeting processing failed: {e}"})


@mcp.tool()
def schedule_meeting_tasks(
    meeting_name: str,
    credentials_path: str = "~/credentials/google_calendar.json",
    preview_only: bool = False,
) -> str:
    """
    Schedule extracted meeting tasks as Google Calendar events.

    Args:
        meeting_name:       Name of the processed meeting (use list_meetings() to find it).
        credentials_path:   Path to the Google Calendar OAuth credentials JSON file.
                            Defaults to ~/credentials/google_calendar.json.
        preview_only:       If True, returns the prepared events without creating them
                            in Google Calendar (useful when credentials are not available).

    Returns JSON with scheduled/failed/skipped counts, or the event list in preview mode.
    """
    cred_path = str(Path(credentials_path).expanduser())

    try:
        task_parser = MeetingTaskParser(meeting_name=meeting_name)
    except Exception as e:
        return json.dumps({"error": f"Failed to load tasks: {e}"})

    if not task_parser.tasks:
        return json.dumps({
            "error": f"No tasks found for meeting '{meeting_name}'. "
                     "Run process_meeting() first, or check the meeting name with list_meetings()."
        })

    try:
        events = task_parser.prepare_calendar_events()
    except Exception as e:
        return json.dumps({"error": f"Failed to prepare calendar events: {e}"})

    # Preview mode — return events without scheduling
    if preview_only or not Path(cred_path).exists():
        preview_events = []
        for ev in events:
            preview_events.append({
                "summary":     ev.get("summary"),
                "start":       ev.get("start", {}).get("dateTime"),
                "end":         ev.get("end",   {}).get("dateTime"),
                "location":    ev.get("location"),
                "description": ev.get("description", "")[:200],
            })
        return json.dumps({
            "mode":     "preview",
            "events":   preview_events,
            "message":  (
                "Preview mode: events NOT created in Google Calendar. "
                "Provide valid credentials_path and set preview_only=false to schedule."
                if not preview_only
                else "Preview mode as requested."
            ),
        }, indent=2)

    # Live scheduling (non-interactive)
    try:
        scheduler = CalendarScheduler(credentials_path=cred_path)
        results = scheduler.schedule_all_events(events, interactive=False)
        return json.dumps({
            "mode":      "scheduled",
            "scheduled": results.get("scheduled", 0),
            "failed":    results.get("failed",    0),
            "skipped":   results.get("skipped",   0),
            "total":     len(events),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Scheduling failed: {e}"})


# ============================================================================
# Email Tools (only if email service available)
# ============================================================================

if EMAIL_SERVICE_AVAILABLE and email_service:
    @mcp.tool()
    def list_emails() -> str:
        """List all processed emails."""
        emails = email_service.list_emails()
        return json.dumps(emails, indent=2)

    @mcp.tool()
    def fetch_emails_batch(max_results: int = 50, label_ids: str = "INBOX") -> str:
        """Fetch recent emails from Gmail."""
        label_list = [l.strip() for l in label_ids.split(",")] if label_ids else None
        try:
            result = email_service.fetch_emails_batch(max_results=max_results, label_ids=label_list)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def rebuild_email_collection() -> str:
        """Rebuild the merged email collection."""
        try:
            result = email_service.rebuild_collection()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def get_email_info(email_id: str) -> str:
        """Get metadata for a specific email."""
        try:
            info = email_service.get_email_info(email_id)
            return json.dumps(info, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def get_email_markdown(email_id: str) -> str:
        """Return the full markdown content of an email."""
        try:
            return email_service.get_email_markdown(email_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def get_email_graph_stats(email_id: str) -> str:
        """Get graph node/edge counts for a vectorized email."""
        try:
            stats = email_service.get_email_graph_stats(email_id)
            return json.dumps(stats, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def query_email_collection(query: str, top_k: int = 5) -> str:
        """Query the merged email collection."""
        try:
            result = email_service.query_email_collection(query, top_k=top_k)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
else:
    # Stub tools with error messages if email service not available
    @mcp.tool()
    def list_emails() -> str:
        return json.dumps({"error": "Email service not available"})

    @mcp.tool()
    def fetch_emails_batch(max_results: int = 50, label_ids: str = "INBOX") -> str:
        return json.dumps({"error": "Email service not available"})

    @mcp.tool()
    def rebuild_email_collection() -> str:
        return json.dumps({"error": "Email service not available"})

    @mcp.tool()
    def get_email_info(email_id: str) -> str:
        return json.dumps({"error": "Email service not available"})

    @mcp.tool()
    def get_email_markdown(email_id: str) -> str:
        return json.dumps({"error": "Email service not available"})

    @mcp.tool()
    def get_email_graph_stats(email_id: str) -> str:
        return json.dumps({"error": "Email service not available"})

    @mcp.tool()
    def query_email_collection(query: str, top_k: int = 5) -> str:
        return json.dumps({"error": "Email service not available"})


# ============================================================================
# Optional MCP Resources
# ============================================================================

@mcp.resource("document://list")
def resource_document_list() -> str:
    """Resource: list of document IDs and short info (JSON)."""
    documents = []
    if document_service.OUTPUT_DIR.exists():
        for doc_dir in document_service.OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                info = document_service.get_document_info(doc_dir.name)
                if info:
                    documents.append({
                        "document_id":  info["document_id"],
                        "name":         info["name"],
                        "status":       info["status"],
                        "total_pages":  info.get("total_pages"),
                        "total_chunks": info.get("total_chunks"),
                    })
    return json.dumps(documents, indent=2)


@mcp.resource("document://{document_id}/markdown")
def resource_document_markdown(document_id: str) -> str:
    """Resource: full markdown content for one document."""
    try:
        return document_service.get_document_markdown(document_id)
    except (ValueError, Exception) as e:
        return json.dumps({"error": str(e)})


@mcp.resource("document://{document_id}/info")
def resource_document_info(document_id: str) -> str:
    """Resource: JSON metadata for one document."""
    info = document_service.get_document_info(document_id)
    if not info:
        return json.dumps({"error": f"Document '{document_id}' not found"})
    return json.dumps(info, indent=2)


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Document Processing Pipeline MCP server (stdio or Streamable HTTP)."
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help=(
            "Run with Streamable HTTP transport so MCP Inspector can connect at "
            "http://127.0.0.1:%s/mcp" % MCP_HTTP_PORT
        ),
    )
    args = parser.parse_args()

    if args.http:
        print(
            "MCP server (Streamable HTTP) starting at "
            "http://127.0.0.1:%s/mcp" % MCP_HTTP_PORT,
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http")
    else:
        import logging
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.root.addHandler(h)
        logging.root.setLevel(logging.INFO)
        mcp.run()


if __name__ == "__main__":
    main()