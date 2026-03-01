"""
FastAPI Application for Document Processing Pipeline
Integrates PDF detection, vectorization, retrieval, and visualization
"""

import asyncio
import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

import networkx as nx
import uvicorn
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Query,
    Path as PathParam,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import modules
from config.inference_config import check_inference_ready
from visualizeGraphE import (
    load_graph,
    visualize_interactive,
    visualize_static,
    visualize_simplified,
    find_graph_file as find_viz_graph_file,
)
from economics_tracker import log_upload, get_usage_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - use service layer for paths and document ops
from services import document_service

OUTPUT_DIR = document_service.OUTPUT_DIR
UPLOAD_DIR = document_service.UPLOAD_DIR
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- PYDANTIC MODELS ----------------


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_available: bool
    ollama_models: List[str]


class DocumentInfo(BaseModel):
    """Document information"""
    document_id: str
    name: str
    status: str  # "uploaded", "processing", "vectorized", "ready"
    # Paths to artifacts
    markdown_path: Optional[str] = None
    page_mapping_path: Optional[str] = None
    vector_mapping_path: Optional[str] = None
    graph_path: Optional[str] = None
    vector_db_path: Optional[str] = None
    confidence_path: Optional[str] = None
    # Summary stats
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None
    # Docling confidence (document-level)
    layout_score: Optional[float] = None
    ocr_score: Optional[float] = None
    parse_score: Optional[float] = None
    table_score: Optional[float] = None
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None


class ConfidencePageScores(BaseModel):
    """Per-page confidence scores"""
    parse_score: Optional[float] = None
    layout_score: Optional[float] = None
    table_score: Optional[float] = None
    ocr_score: Optional[float] = None


class ConfidenceResponse(BaseModel):
    """Confidence / accuracy information for a document"""
    document_id: str
    has_confidence: bool
    layout_score: Optional[float] = None
    ocr_score: Optional[float] = None
    parse_score: Optional[float] = None
    table_score: Optional[float] = None
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None
    pages: Dict[str, ConfidencePageScores] = Field(default_factory=dict)


class ProcessPDFRequest(BaseModel):
    """Request to process a PDF"""
    document_id: Optional[str] = None
    preserve_page_breaks: bool = True


class ProcessPDFResponse(BaseModel):
    """Response from PDF processing"""
    document_id: str
    status: str
    message: str
    markdown_path: Optional[str] = None
    page_mapping_path: Optional[str] = None
    total_pages: Optional[int] = None


class VectorizeRequest(BaseModel):
    """Request to vectorize a document"""
    document_id: str


class VectorizeResponse(BaseModel):
    """Response from vectorization"""
    document_id: str
    status: str
    message: str
    vector_mapping_path: Optional[str] = None
    graph_path: Optional[str] = None
    vector_db_path: Optional[str] = None
    total_chunks: Optional[int] = None
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None


class ChunkDetail(BaseModel):
    """Detailed information about a retrieved chunk"""
    chunk_index: int
    content: str
    heading: str
    section_path: str
    section_title: str
    page_number: Optional[int] = None
    page_classification: Optional[str] = None
    summary: str
    chunk_type: str
    has_table: bool
    table_context: Optional[str] = None
    start_line: Optional[int] = None
    content_length: int
    retrieval_source: str = Field(..., description="How chunk was retrieved: 'seed', 'graph_expanded', 'second_seed', 'second_expanded', 'initial', 'second_retrieval'")
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = Field(None, description="Re-ranking relevance score [0-1]")


class QueryRequest(BaseModel):
    """Query request"""
    query: str = Field(..., description="The query to search for")
    document_id: str = Field(..., description="Document ID to query")
    include_chunks: bool = Field(default=True, description="Include detailed chunk information in response")


class QueryResponse(BaseModel):
    """Query response with detailed chunk information"""
    answer: str
    document_id: str
    query: str
    retrieval_stats: Dict[str, Any]
    chunk_analysis: Optional[str] = None
    chunks: List[ChunkDetail] = Field(default_factory=list, description="Detailed information about chunks used to generate the answer")
    debug_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Debugging information about retrieval process")


class VisualizationRequest(BaseModel):
    """Request for graph visualization"""
    document_id: str
    visualization_type: str = Field(
        default="interactive",
        description="Type: interactive, static, simplified"
    )


class PageSummaryRequest(BaseModel):
    """Request for page summarization"""
    document_id: str = Field(..., description="Document ID")
    page_number: int = Field(..., description="Page number to summarize", gt=0)


class PageSummaryResponse(BaseModel):
    """Page summary response"""
    page_number: int
    summary: str
    key_points: List[str]
    sections: List[str]
    has_content: bool
    used_adjacent_pages: bool
    adjacent_pages_used: Optional[List[int]] = None
    page_classification: Optional[str] = None
    chunks_used: List[int]
    total_chunks: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


# ---------------- LIFECYCLE MANAGEMENT ----------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")
    
    # Check inference backend on startup
    is_running, models = check_inference_ready()
    if not is_running:
        logger.warning("Inference backend not detected. Some features may not work.")
    else:
        logger.info("Inference backend ready: %s", models[:5] if len(models) > 5 else models)
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    _loaded_agents.clear()


# ---------------- FASTAPI APP ----------------


app = FastAPI(
    title="Document Processing API",
    description="API for PDF processing, vectorization, graph-based retrieval, and visualization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- HELPER FUNCTIONS (delegate to service) ----------------


def get_document_path(document_id: str) -> Path:
    """Get document output directory"""
    return document_service.get_document_path(document_id)


def get_document_info(document_id: str) -> Optional[DocumentInfo]:
    """Get document information"""
    info = document_service.get_document_info(document_id)
    return DocumentInfo(**info) if info else None


def load_agent_for_document(document_id: str) -> Dict[str, Any]:
    """Load agent resources for a document. Raises HTTPException if not vectorized."""
    try:
        return document_service.load_agent_for_document(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------- API ENDPOINTS ----------------


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "documents": "/documents",
            "upload": "/upload",
            "vectorize": "/vectorize/{document_id}",
            "query": "/query",
            "visualize": "/visualize/{document_id}",
            "page_summarize": "/page/{document_id}/summarize",
            "economics_summary": "/economics/summary",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint (inference backend: Ollama or Hugging Face)"""
    is_running, models = check_inference_ready()
    return HealthResponse(
        status="healthy" if is_running else "degraded",
        ollama_available=is_running,
        ollama_models=models,
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all processed documents"""
    documents = []
    if OUTPUT_DIR.exists():
        for doc_dir in OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                doc_info = get_document_info(doc_dir.name)
                if doc_info:
                    documents.append(doc_info)
    return documents


@app.get("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(document_id: str = PathParam(..., description="Document ID")):
    """Get document information"""
    doc_info = get_document_info(document_id)
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    return doc_info


@app.get("/documents/{document_id}/markdown", tags=["Documents"])
async def get_markdown(document_id: str = PathParam(..., description="Document ID")):
    """Get markdown content for a document"""
    try:
        content = document_service.get_document_markdown(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    from fastapi.responses import Response
    return Response(content=content, media_type="text/plain")


@app.get("/documents/{document_id}/confidence", response_model=ConfidenceResponse, tags=["Documents"])
async def get_confidence(document_id: str = PathParam(..., description="Document ID")):
    """Get Docling confidence / accuracy scores for a document"""
    doc_info = get_document_info(document_id)
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )

    if not doc_info.confidence_path:
        # No confidence available (older runs or failure)
        return ConfidenceResponse(
            document_id=document_id,
            has_confidence=False,
        )

    conf_path = Path(doc_info.confidence_path)
    if not conf_path.exists():
        return ConfidenceResponse(
            document_id=document_id,
            has_confidence=False,
        )

    try:
        with open(conf_path, "r", encoding="utf-8") as f:
            conf_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read confidence file for {document_id}: {e}", exc_info=True)
        return ConfidenceResponse(
            document_id=document_id,
            has_confidence=False,
        )

    # Build per-page scores map
    pages_raw = conf_data.get("pages") or {}
    pages: Dict[str, ConfidencePageScores] = {}
    for page_no, scores in pages_raw.items():
        # keys might be ints or strings; normalize to string for frontend
        key = str(page_no)
        pages[key] = ConfidencePageScores(
            parse_score=scores.get("parse_score"),
            layout_score=scores.get("layout_score"),
            table_score=scores.get("table_score"),
            ocr_score=scores.get("ocr_score"),
        )

    return ConfidenceResponse(
        document_id=document_id,
        has_confidence=True,
        layout_score=conf_data.get("layout_score"),
        ocr_score=conf_data.get("ocr_score"),
        parse_score=conf_data.get("parse_score"),
        table_score=conf_data.get("table_score"),
        mean_grade=conf_data.get("mean_grade"),
        low_grade=conf_data.get("low_grade"),
        pages=pages,
    )


@app.post("/upload", response_model=ProcessPDFResponse, tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload and process a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Generate document ID
    document_id = str(uuid4())
    doc_path = get_document_path(document_id)
    doc_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    pdf_path = doc_path / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_size = pdf_path.stat().st_size
    log_upload(document_id, file_size_bytes=file_size, filename=file.filename or "")
    
    # Process in background
    background_tasks.add_task(process_pdf_background, str(pdf_path), document_id)
    
    return ProcessPDFResponse(
        document_id=document_id,
        status="processing",
        message="PDF uploaded and processing started",
    )


def process_pdf_background(pdf_path: str, document_id: str):
    """Background task to process PDF (file already saved in document folder)"""
    try:
        logger.info(f"Processing PDF: {pdf_path} for document {document_id}")
        document_service.run_detection_for_document(document_id)
        logger.info(f"PDF processing complete for document {document_id}")
    except ValueError as e:
        logger.error(f"Error processing PDF {document_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)


@app.post("/vectorize/{document_id}", response_model=VectorizeResponse, tags=["Processing"])
async def vectorize_document(
    document_id: str = PathParam(..., description="Document ID"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Vectorize a processed document"""
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    
    # Check if markdown exists
    md_files = list(doc_path.glob("*.md"))
    if not md_files:
        raise HTTPException(
            status_code=400,
            detail="Document must be processed first (markdown not found)"
        )
    
    # Process in background
    background_tasks.add_task(vectorize_background, document_id)
    
    return VectorizeResponse(
        document_id=document_id,
        status="processing",
        message="Vectorization started",
    )


def vectorize_background(document_id: str):
    """Background task to vectorize document"""
    try:
        logger.info(f"{'='*80}")
        logger.info(f"🚀 Starting vectorization for document: {document_id}")
        logger.info(f"{'='*80}")
        document_service.trigger_vectorize(document_id)
        doc_path = get_document_path(document_id)
        logger.info(f"{'='*80}")
        logger.info(f"✅ Vectorization complete for document {document_id}")
        logger.info(f"   📂 Output directory: {doc_path / 'E'}")
        logger.info(f"{'='*80}")
    except ValueError as e:
        logger.error(f"❌ Vectorization failed for document {document_id}: {e}")
    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"❌ Error vectorizing document {document_id}: {e}")
        logger.error(f"{'='*80}")
        logger.error(f"Error details:", exc_info=True)


@app.get("/economics/summary", tags=["Economics"])
async def economics_summary(date: Optional[str] = Query(None, description="Date YYYY-MM-DD (default: today UTC)")):
    """
    Token usage and cost visibility for stakeholders.
    Returns aggregated token counts by phase (upload, vectorization, retrieval, page_summary)
    and by step. Data is read from the economics/ folder (usage_<date>.jsonl).
    """
    try:
        summary = get_usage_summary(date_str=date)
        return summary
    except Exception as e:
        logger.error(f"Economics summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_document(request: QueryRequest):
    """Query a vectorized document using graph-enhanced retrieval"""
    try:
        result = document_service.query_document(
            request.document_id,
            request.query,
            include_chunks=request.include_chunks,
        )
    except ValueError as e:
        raise HTTPException(status_code=400 if "query" in str(e).lower() else 404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Build ChunkDetail list from service dicts
    chunks_detail = [
        ChunkDetail(
            chunk_index=c["chunk_index"],
            content=c["content"],
            heading=c["heading"],
            section_path=c["section_path"],
            section_title=c["section_title"],
            page_number=c.get("page_number"),
            page_classification=c.get("page_classification"),
            summary=c["summary"],
            chunk_type=c["chunk_type"],
            has_table=c["has_table"],
            table_context=c.get("table_context"),
            start_line=c.get("start_line"),
            content_length=c["content_length"],
            retrieval_source=c["retrieval_source"],
            similarity_score=c.get("similarity_score"),
            rerank_score=c.get("rerank_score"),
        )
        for c in result["chunks"]
    ]
    return QueryResponse(
        answer=result["answer"],
        document_id=result["document_id"],
        query=result["query"],
        retrieval_stats=result["retrieval_stats"],
        chunk_analysis=result.get("chunk_analysis"),
        chunks=chunks_detail,
        debug_info=result.get("debug_info") or {},
    )


@app.post("/visualize/{document_id}", tags=["Visualization"])
async def visualize_graph(
    document_id: str = PathParam(..., description="Document ID"),
    viz_type: str = Query(default="interactive", description="Type: interactive, static, simplified"),
):
    """Generate graph visualization"""
    doc_path = get_document_path(document_id)
    plan_e_dir = doc_path / "E"
    
    if not plan_e_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not vectorized. Run vectorization first."
        )
    
    # Find graph file
    graph_file = find_viz_graph_file(plan_e_dir)
    if not graph_file or not graph_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Graph file not found"
        )
    
    # Load graph
    try:
        G = load_graph(graph_file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load graph: {str(e)}"
        )
    
    # Create output directory
    viz_dir = plan_e_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    graph_name = graph_file.stem.replace("_document_graph", "")
    
    try:
        if viz_type == "interactive":
            output_file = viz_dir / f"{graph_name}_interactive.html"
            visualize_interactive(G, output_file)
            return FileResponse(
                output_file,
                media_type="text/html",
                filename=output_file.name,
            )
        elif viz_type == "static":
            output_file = viz_dir / f"{graph_name}_full.png"
            visualize_static(G, output_file, layout="spring")
            return FileResponse(
                output_file,
                media_type="image/png",
                filename=output_file.name,
            )
        elif viz_type == "simplified":
            output_file = viz_dir / f"{graph_name}_simplified.png"
            visualize_simplified(G, output_file, max_nodes=150)
            return FileResponse(
                output_file,
                media_type="image/png",
                filename=output_file.name,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown visualization type: {viz_type}"
            )
    except Exception as e:
        logger.error(f"Error creating visualization: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Visualization failed: {str(e)}"
        )


@app.get("/visualize/{document_id}/stats", tags=["Visualization"])
async def get_graph_stats(document_id: str = PathParam(..., description="Document ID")):
    """Get graph statistics"""
    try:
        return document_service.get_graph_stats(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/page/{document_id}/summarize", response_model=PageSummaryResponse, tags=["Page Analysis"])
async def summarize_page(
    document_id: str = PathParam(..., description="Document ID"),
    page_number: int = Query(..., description="Page number to summarize", gt=0),
):
    """Generate comprehensive page-level summary and explanation"""
    try:
        result = document_service.summarize_page(document_id, page_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error summarizing page {page_number} for document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return PageSummaryResponse(
        page_number=result["page_number"],
        summary=result["summary"],
        key_points=result["key_points"],
        sections=result["sections"],
        has_content=result["has_content"],
        used_adjacent_pages=result["used_adjacent_pages"],
        adjacent_pages_used=result.get("adjacent_pages_used"),
        page_classification=result.get("page_classification"),
        chunks_used=result["chunks_used"],
        total_chunks=result["total_chunks"],
    )


# ---------------- ERROR HANDLERS ----------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


# ---------------- MAIN ----------------


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
