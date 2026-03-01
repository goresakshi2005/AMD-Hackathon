"""
RetrievalAgentE - Plan E Implementation: LangGraph Agent with Graph-Enhanced Retrieval
Uses LangGraph to create an agent that uses graph traversal for intelligent retrieval.
Expands from seed chunks via graph edges to capture document relationships.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.error("networkx is required. Install with: pip install networkx")
    sys.exit(1)

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from config.inference_config import get_embeddings, get_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

# ---------------- DOCUMENT GRAPH ---------------- 
class DocumentGraph:
    """Knowledge graph for document structure"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.chunk_nodes = {}
        self.section_nodes = {}
        self.page_nodes = {}
    
    def get_parent_section(self, chunk_id: int) -> Optional[str]:
        """Get the parent section node for a chunk"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return None
        
        # Find section node connected via "belongs_to"
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "section":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "belongs_to":
                    return predecessor
        
        return None
    
    def get_section_chunks(self, section_path: str) -> List[int]:
        """Get all chunk IDs in a section"""
        section_node = self.section_nodes.get(section_path)
        if not section_node:
            return []
        
        chunk_ids = []
        for successor in self.graph.successors(section_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(section_node, successor)
                if edge_data and edge_data.get("relation") == "contains":
                    chunk_id = self.graph.nodes[successor].get("chunk_id")
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
        
        return sorted(chunk_ids)
    
    def get_adjacent_chunks(self, chunk_id: int, window: int = 1) -> List[int]:
        """Get adjacent chunks via 'follows' edges"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return []
        
        adjacent_ids = []
        
        # Get previous chunks (incoming "follows" edges)
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "follows":
                    prev_id = self.graph.nodes[predecessor].get("chunk_id")
                    if prev_id is not None:
                        adjacent_ids.append(prev_id)
        
        # Get next chunks (outgoing "follows" edges)
        for successor in self.graph.successors(chunk_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(chunk_node, successor)
                if edge_data and edge_data.get("relation") == "follows":
                    next_id = self.graph.nodes[successor].get("chunk_id")
                    if next_id is not None:
                        adjacent_ids.append(next_id)
        
        return adjacent_ids[:window * 2]
    
    def get_similar_chunks(self, chunk_id: int) -> List[int]:
        """Get chunks connected via 'similar_to' edges"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return []
        
        similar_ids = []
        
        # Get chunks connected via similarity edges (both directions)
        for successor in self.graph.successors(chunk_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(chunk_node, successor)
                if edge_data and edge_data.get("relation") == "similar_to":
                    similar_id = self.graph.nodes[successor].get("chunk_id")
                    if similar_id is not None:
                        similar_ids.append(similar_id)
        
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "similar_to":
                    similar_id = self.graph.nodes[predecessor].get("chunk_id")
                    if similar_id is not None:
                        similar_ids.append(similar_id)
        
        return similar_ids
    
    def get_page_chunks(self, page_number: int) -> List[int]:
        """Get all chunk IDs on a page"""
        page_node = self.page_nodes.get(page_number)
        if not page_node:
            return []
        
        chunk_ids = []
        for successor in self.graph.successors(page_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(page_node, successor)
                if edge_data and edge_data.get("relation") == "on_page":
                    chunk_id = self.graph.nodes[successor].get("chunk_id")
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
        
        return sorted(chunk_ids)
    
    def expand_from_chunks(self, chunk_ids: List[int], max_expansion: int = 30) -> List[int]:
        """Expand retrieval by following graph edges from seed chunks"""
        expanded_ids = set(chunk_ids)
        to_process = list(chunk_ids)
        
        while to_process and len(expanded_ids) < max_expansion:
            current_id = to_process.pop(0)
            
            # Get parent section and all chunks in that section
            section_node = self.get_parent_section(current_id)
            if section_node:
                section_path = self.graph.nodes[section_node].get("section_path", "")
                section_chunks = self.get_section_chunks(section_path)
                for chunk_id in section_chunks:
                    if chunk_id not in expanded_ids:
                        expanded_ids.add(chunk_id)
                        to_process.append(chunk_id)
            
            # Get adjacent chunks
            adjacent = self.get_adjacent_chunks(current_id, window=1)
            for adj_id in adjacent:
                if adj_id not in expanded_ids:
                    expanded_ids.add(adj_id)
                    to_process.append(adj_id)
            
            # Get similar chunks via similarity edges
            similar = self.get_similar_chunks(current_id)
            for sim_id in similar:
                if sim_id not in expanded_ids:
                    expanded_ids.add(sim_id)
                    to_process.append(sim_id)
        
        return list(expanded_ids)
    
    def load(self, filepath: Path):
        """Load graph from JSON format"""
        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph.clear()
        self.chunk_nodes.clear()
        self.section_nodes.clear()
        self.page_nodes.clear()
        
        # Add nodes
        for node_data in graph_data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)
            
            # Rebuild index
            if node_data.get("type") == "chunk":
                chunk_id = node_data.get("chunk_id")
                if chunk_id is not None:
                    self.chunk_nodes[chunk_id] = node_id
            elif node_data.get("type") == "section":
                section_path = node_data.get("section_path", "")
                if section_path:
                    self.section_nodes[section_path] = node_id
            elif node_data.get("type") == "page":
                page_number = node_data.get("page_number")
                if page_number:
                    self.page_nodes[page_number] = node_id
        
        # Add edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        
        logger.info(f"Loaded graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

# ---------------- STATE DEFINITION ---------------- 
class AgentState(TypedDict):
    """State for the retrieval agent workflow"""
    query: str
    is_page_summary: bool  # Whether query is asking for page summary
    page_number: Optional[int]  # Page number if is_page_summary is True
    seed_chunk_ids: List[int]  # Initial vector search results
    seed_chunk_scores: Dict[int, float]  # Similarity scores for seed chunks
    graph_expanded_ids: List[int]  # Chunks expanded via graph
    retrieved_chunks: List[Document]  # All retrieved chunks
    reranked_chunks: Optional[List[Document]]  # Re-ranked chunks by relevance
    rerank_scores: Optional[Dict[int, float]]  # Re-ranking scores
    chunk_analysis: str
    needs_more_info: bool
    new_query: Optional[str]
    second_seed_ids: List[int]
    second_seed_scores: Dict[int, float]  # Similarity scores for second seed chunks
    second_expanded_ids: List[int]
    second_retrieval_chunks: List[Document]
    final_answer: str
    iteration_count: int
    document_folder: Optional[str]  # Path to document folder for page summarization
    debug_info: Dict[str, Any]  # Debugging information
    token_usage: Optional[List[Dict[str, Any]]]  # Per-step token counts for economics reporting

# ---------------- TOKEN USAGE HELPERS (economics) ----------------
def _est_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(str(text).strip()) // 4)

def _append_token_usage(
    state: AgentState,
    step: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one step's token usage to state for economics logging."""
    usage = state.get("token_usage") or []
    rec = {"step": step, "input_tokens": input_tokens, "output_tokens": output_tokens, "embedding_tokens": embedding_tokens}
    if extra:
        rec["extra"] = extra
    usage.append(rec)
    state["token_usage"] = usage

# ---------------- LOAD FUNCTIONS ---------------- 
def load_chunks_from_mapping(mapping_file: Path) -> List[Document]:
    """Load chunks from vector mapping JSON file"""
    if not mapping_file.exists():
        logger.error(f"Vector mapping file not found: {mapping_file}")
        return []
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    chunks = []
    for item in mapping_data:
        doc = Document(
            page_content=item.get("content", ""),
            metadata=item.get("metadata", {})
        )
        chunks.append(doc)
    
    logger.info(f"Loaded {len(chunks)} chunks from mapping file")
    return chunks

def load_vector_store(vector_db_path: Path) -> Chroma:
    """Load Chroma vector store (uses inference_config for embeddings)."""
    if not vector_db_path.exists():
        logger.error(f"Vector DB path not found: {vector_db_path}")
        sys.exit(1)
    embeddings = get_embeddings()
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_path)
    )
    logger.info(f"Loaded vector store from: {vector_db_path}")
    return vector_store

# Global stores
_vector_store: Optional[Chroma] = None
_document_graph: Optional[DocumentGraph] = None
_chunks: List[Document] = []
_llm: Optional[Any] = None  # BaseChatModel (Ollama or Hugging Face)
_page_agent: Optional[Any] = None  # PageSummarizationAgent
_document_folder: Optional[Path] = None

def set_agent_resources(vector_store: Chroma, document_graph: DocumentGraph, chunks: List[Document], llm: Any, document_folder: Optional[Path] = None):
    """Set global resources for agent nodes"""
    global _vector_store, _document_graph, _chunks, _llm, _page_agent, _document_folder
    _vector_store = vector_store
    _document_graph = document_graph
    _chunks = chunks
    _llm = llm
    _document_folder = document_folder
    
    # Load page summarization agent if document folder is provided (reuse same LLM)
    if document_folder:
        try:
            from page_summarization import load_page_agent
            _page_agent = load_page_agent(document_folder, llm=llm)
            if _page_agent:
                logger.info("Page summarization agent loaded successfully")
            else:
                logger.warning("Failed to load page summarization agent")
        except Exception as e:
            logger.warning(f"Could not load page summarization agent: {e}")
            _page_agent = None

# ---------------- QUERY CLASSIFICATION ---------------- 
def classify_query(state: AgentState) -> AgentState:
    """Classify query to determine if it's asking for page summary or normal retrieval"""
    query = state["query"].lower()
    llm = _llm
    
    logger.info("Classifying query type...")
    
    # Quick pattern matching for common page summary queries
    page_patterns = [
        r'page\s+(\d+)',
        r'page\s+number\s+(\d+)',
        r'summarize\s+page\s+(\d+)',
        r'summary\s+of\s+page\s+(\d+)',
        r'what\s+is\s+on\s+page\s+(\d+)',
        r'content\s+of\s+page\s+(\d+)',
        r'explain\s+page\s+(\d+)',
    ]
    
    page_number = None
    for pattern in page_patterns:
        match = re.search(pattern, query)
        if match:
            try:
                page_number = int(match.group(1))
                logger.info(f"Detected page number from pattern: {page_number}")
                state["is_page_summary"] = True
                state["page_number"] = page_number
                return state
            except ValueError:
                continue
    
    # Use LLM for more complex classification
    classification_prompt = f"""Is this query asking for a page summary (specific page number) or a general document question?

Query: {state["query"]}

Reply EXACTLY:
IS_PAGE_SUMMARY: [YES or NO]
PAGE_NUMBER: [number if YES, else None]"""
    
    try:
        response = llm.invoke(classification_prompt)
        response_text = (getattr(response, "content", None) or str(response)).strip()
        _append_token_usage(
            state, "query_classification",
            input_tokens=_est_tokens(classification_prompt),
            output_tokens=_est_tokens(response_text),
        )
        
        is_page_summary = False
        page_num = None
        
        if "IS_PAGE_SUMMARY:" in response_text.upper():
            summary_line = [line for line in response_text.split("\n") if "IS_PAGE_SUMMARY:" in line.upper()]
            if summary_line:
                summary_value = summary_line[0].split(":")[-1].strip().upper()
                is_page_summary = summary_value == "YES"
        
        if is_page_summary and "PAGE_NUMBER:" in response_text.upper():
            page_line = [line for line in response_text.split("\n") if "PAGE_NUMBER:" in line.upper()]
            if page_line:
                page_value = page_line[0].split(":")[-1].strip()
                try:
                    page_num = int(page_value)
                except ValueError:
                    # Try to extract number from the value
                    num_match = re.search(r'\d+', page_value)
                    if num_match:
                        page_num = int(num_match.group())
        
        state["is_page_summary"] = is_page_summary
        state["page_number"] = page_num if is_page_summary else None
        
        if is_page_summary and page_num:
            logger.info(f"Query classified as page summary request for page {page_num}")
        else:
            logger.info("Query classified as normal retrieval request")
        
    except Exception as e:
        logger.error(f"Error classifying query: {e}")
        state["is_page_summary"] = False
        state["page_number"] = None
    
    return state

# ---------------- PAGE SUMMARIZATION ---------------- 
def summarize_page(state: AgentState) -> AgentState:
    """Generate page summary using PageSummarizationAgent"""
    page_number = state.get("page_number")
    page_agent = _page_agent
    
    if not page_number:
        state["final_answer"] = "Error: No page number specified for page summary."
        return state
    
    if not page_agent:
        state["final_answer"] = f"Error: Page summarization is not available. Cannot summarize page {page_number}."
        return state
    
    logger.info(f"Generating summary for page {page_number}...")
    
    try:
        page_summary = page_agent.summarize_page(page_number, use_adjacent_if_empty=True)
        
        # Format the response
        answer_parts = []
        
        if page_summary.page_classification:
            answer_parts.append(f"**Page {page_number} Classification:** {page_summary.page_classification}\n")
        
        answer_parts.append(f"**Summary:**\n{page_summary.summary}\n")
        
        if page_summary.key_points:
            answer_parts.append("**Key Points:**")
            for point in page_summary.key_points:
                answer_parts.append(f"- {point}")
            answer_parts.append("")
        
        if page_summary.sections:
            answer_parts.append(f"**Sections:** {', '.join(page_summary.sections)}")
        
        if page_summary.used_adjacent_pages and page_summary.adjacent_pages_used:
            answer_parts.append(f"\n*Note: This summary is based on adjacent pages {page_summary.adjacent_pages_used} as page {page_number} had limited direct content.*")
        
        state["final_answer"] = "\n".join(answer_parts)
        logger.info(f"Page summary generated successfully for page {page_number}")
        
    except Exception as e:
        logger.error(f"Error generating page summary: {e}", exc_info=True)
        state["final_answer"] = f"Error generating summary for page {page_number}: {str(e)}"
    
    return state

# ---------------- RETRIEVAL FUNCTIONS ---------------- 
def distance_to_similarity(distance: float, scale_factor: float = 150.0) -> float:
    """Convert L2 distance to similarity score [0, 1]"""
    similarity = 1.0 / (1.0 + (distance / scale_factor))
    return max(0.0, min(1.0, similarity))

def initial_retrieval(state: AgentState) -> AgentState:
    """Initial retrieval: vector search + selective graph expansion"""
    query = state["query"]
    vector_store = _vector_store
    document_graph = _document_graph
    
    logger.info(f"Initial graph-enhanced retrieval for query: '{query}'")
    
    # Step 1: Vector similarity search (get seed chunks) - increased k for better recall
    seed_chunk_ids = []
    seed_chunk_scores = {}
    seed_chunks_with_scores = []
    
    try:
        vector_results = vector_store.similarity_search_with_score(query, k=20)  # Increased from 8 to 20
        for doc, distance in vector_results:
            chunk_id = doc.metadata.get("chunk_index")
            if chunk_id is not None:
                similarity_score = distance_to_similarity(distance)
                seed_chunks_with_scores.append((chunk_id, similarity_score, distance))
                seed_chunk_ids.append(chunk_id)
                seed_chunk_scores[chunk_id] = similarity_score
                logger.debug(f"  Seed chunk {chunk_id} (distance: {distance:.3f}, similarity: {similarity_score:.3f})")
    except Exception as e:
        logger.warning(f"Vector search error: {e}")
    
    logger.info(f"Found {len(seed_chunk_ids)} seed chunks from vector search")
    
    # Economics: query embedding for vector search
    _append_token_usage(state, "initial_retrieval", embedding_tokens=_est_tokens(query))
    
    # Step 2: Filter seed chunks by similarity threshold and select top-k for expansion
    SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to consider
    TOP_SEEDS_FOR_EXPANSION = 5  # Only expand from top 5 most relevant seeds
    
    # Sort by similarity score (descending)
    seed_chunks_with_scores.sort(reverse=True, key=lambda x: x[1])
    
    # Filter by threshold
    high_quality_seeds = [chunk_id for chunk_id, score, _ in seed_chunks_with_scores if score >= SIMILARITY_THRESHOLD]
    
    # Select top-k seeds for graph expansion (only most relevant)
    top_seeds_for_expansion = [chunk_id for chunk_id, _, _ in seed_chunks_with_scores[:TOP_SEEDS_FOR_EXPANSION]]
    
    logger.info(f"High quality seeds (score >= {SIMILARITY_THRESHOLD}): {len(high_quality_seeds)}")
    logger.info(f"Top {TOP_SEEDS_FOR_EXPANSION} seeds selected for graph expansion: {top_seeds_for_expansion}")
    
    # Step 3: Selective graph expansion - only from top relevant seeds
    graph_expanded_ids = []
    if top_seeds_for_expansion:
        # Use reduced max_expansion since we're being more selective
        graph_expanded_ids = document_graph.expand_from_chunks(top_seeds_for_expansion, max_expansion=15)
        logger.info(f"Graph expansion from top seeds: {len(graph_expanded_ids)} additional chunks")
    else:
        logger.warning("No high-quality seeds for graph expansion")
    
    # Combine seed and expanded (prioritize seed chunks)
    all_chunk_ids = list(set(seed_chunk_ids + graph_expanded_ids))
    logger.info(f"Total chunks after expansion: {len(all_chunk_ids)} (seeds: {len(seed_chunk_ids)}, expanded: {len(graph_expanded_ids)})")
    
    # Get actual Document objects
    retrieved_chunks = []
    chunk_dict = {chunk.metadata.get("chunk_index"): chunk for chunk in _chunks}
    for chunk_id in all_chunk_ids:
        if chunk_id in chunk_dict:
            retrieved_chunks.append(chunk_dict[chunk_id])
    
    # Cap chunks used (no re-ranking)
    MAX_INITIAL_CHUNKS = 25
    chunks_used = retrieved_chunks[:MAX_INITIAL_CHUNKS]
    
    # Update state
    state["seed_chunk_ids"] = seed_chunk_ids
    state["seed_chunk_scores"] = seed_chunk_scores
    state["graph_expanded_ids"] = graph_expanded_ids
    state["retrieved_chunks"] = chunks_used
    state["reranked_chunks"] = chunks_used
    state["rerank_scores"] = {}
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    # Add debug info
    debug_info = state.get("debug_info", {})
    debug_info["initial_retrieval"] = {
        "total_seed_chunks": len(seed_chunk_ids),
        "high_quality_seeds": len(high_quality_seeds),
        "seeds_used_for_expansion": len(top_seeds_for_expansion),
        "graph_expanded_chunks": len(graph_expanded_ids),
        "total_retrieved": len(retrieved_chunks),
        "total_used": len(chunks_used),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "top_seeds_for_expansion": top_seeds_for_expansion,
        "seed_scores": {str(k): round(v, 3) for k, v in list(seed_chunk_scores.items())[:10]},
    }
    state["debug_info"] = debug_info
    
    logger.info(f"Initial retrieval complete: {len(chunks_used)} chunks")
    return state

def analyze_chunks(state: AgentState) -> AgentState:
    """Analyze retrieved chunks to determine if more information is needed"""
    query = state["query"]
    # Use reranked chunks if available (they're already in retrieved_chunks after initial_retrieval)
    retrieved_chunks = state["retrieved_chunks"]
    llm = _llm
    
    logger.info(f"Analyzing {len(retrieved_chunks)} retrieved chunks...")
    
    # Prepare context from chunks (use top chunks, which are already re-ranked)
    chunks_text = []
    for i, chunk in enumerate(retrieved_chunks[:12], 1):
        heading = chunk.metadata.get("heading", "No heading")
        section = chunk.metadata.get("section_path", "No section")
        summary = chunk.metadata.get("summary", "")
        content_preview = chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
        
        chunks_text.append(f"""
Chunk {i}:
- Heading: {heading}
- Section: {section}
- Summary: {summary}
- Content: {content_preview}
""")
    
    context = "\n".join(chunks_text)
    
    # Ask LLM to analyze
    analysis_prompt = f"""Query: {query}

Chunks:
{context}

Do these chunks fully answer the query? If not, give ONE short follow-up search query (5-15 words). Reply EXACTLY:

ANALYSIS: [brief]
MISSING: [what is missing or None]
NEEDS_MORE: [YES or NO]
RELATED_QUERY: [one query or None]"""
    
    try:
        response = llm.invoke(analysis_prompt)
        response_str = (getattr(response, "content", None) or str(response)).strip()
        analysis = response_str
        _append_token_usage(
            state, "analyze_chunks",
            input_tokens=_est_tokens(analysis_prompt),
            output_tokens=_est_tokens(response_str),
        )
        
        # Parse response
        needs_more = False
        new_query = None
        
        if "NEEDS_MORE:" in analysis.upper():
            needs_more_line = [line for line in analysis.split("\n") if "NEEDS_MORE:" in line.upper()]
            if needs_more_line:
                needs_more_value = needs_more_line[0].split(":")[-1].strip().upper()
                needs_more = needs_more_value == "YES"
        
        if needs_more:
            if "RELATED_QUERY:" in analysis.upper():
                query_line = None
                for line in analysis.split("\n"):
                    if "RELATED_QUERY:" in line.upper():
                        query_line = line
                        break
                
                if query_line:
                    query_text = query_line.split(":", 1)[-1].strip()
                    query_text = query_text.strip('"').strip("'").strip()
                    
                    prefixes_to_remove = [
                        "a related query could be",
                        "a query could be",
                        "related query:",
                        "query:",
                        "the query is",
                        "the query should be"
                    ]
                    for prefix in prefixes_to_remove:
                        if query_text.lower().startswith(prefix):
                            query_text = query_text[len(prefix):].strip()
                            query_text = query_text.strip('"').strip("'").strip()
                    
                    if " or " in query_text.lower():
                        query_text = query_text.split(" or ")[0].strip()
                        query_text = query_text.strip('"').strip("'").strip()
                    
                    explanation_markers = ["?", ".", " -", ":", " or", " could", " should"]
                    for marker in explanation_markers:
                        if marker in query_text and query_text.index(marker) < len(query_text) - 10:
                            parts = query_text.split(marker, 1)
                            if len(parts[0]) > 10:
                                query_text = parts[0].strip()
                    
                    query_text = query_text.strip('"').strip("'").strip()
                    query_text = " ".join(query_text.split())
                    
                    invalid_values = ["none", "n/a", "na", "not needed", "sufficient"]
                    if (query_text.lower() not in invalid_values and 
                        len(query_text) > 10 and 
                        len(query_text) < 200):
                        new_query = query_text
                    else:
                        new_query = None
                        needs_more = False
        
        state["chunk_analysis"] = analysis
        state["needs_more_info"] = needs_more
        state["new_query"] = new_query
        
        logger.info(f"Analysis complete. Needs more info: {needs_more}")
        if new_query:
            logger.info(f"New query: {new_query}")
        else:
            logger.info("No new query generated")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        state["chunk_analysis"] = "Analysis failed"
        state["needs_more_info"] = False
        state["new_query"] = None
    
    return state

def second_retrieval(state: AgentState) -> AgentState:
    """Perform second retrieval with new query: vector search + selective graph expansion"""
    new_query = state["new_query"]
    vector_store = _vector_store
    document_graph = _document_graph
    existing_chunk_ids = set(state["seed_chunk_ids"] + state["graph_expanded_ids"])
    
    logger.info(f"Second graph-enhanced retrieval for query: '{new_query}'")
    
    # Step 1: Vector search with new query
    second_seed_ids = []
    second_seed_scores = {}
    second_seeds_with_scores = []
    
    try:
        vector_results = vector_store.similarity_search_with_score(new_query, k=15)  # Increased from 6
        for doc, distance in vector_results:
            chunk_id = doc.metadata.get("chunk_index")
            if chunk_id is not None and chunk_id not in existing_chunk_ids:
                similarity_score = distance_to_similarity(distance)
                second_seeds_with_scores.append((chunk_id, similarity_score, distance))
                second_seed_ids.append(chunk_id)
                second_seed_scores[chunk_id] = similarity_score
                logger.debug(f"  Seed chunk {chunk_id} (distance: {distance:.3f}, similarity: {similarity_score:.3f})")
    except Exception as e:
        logger.warning(f"Vector search error: {e}")
    
    logger.info(f"Found {len(second_seed_ids)} new seed chunks")
    _append_token_usage(state, "second_retrieval", embedding_tokens=_est_tokens(new_query or ""))
    
    # Step 2: Selective graph expansion - only from top relevant seeds
    SIMILARITY_THRESHOLD = 0.3
    TOP_SEEDS_FOR_EXPANSION = 4  # Fewer for second retrieval
    
    second_seeds_with_scores.sort(reverse=True, key=lambda x: x[1])
    high_quality_second_seeds = [chunk_id for chunk_id, score, _ in second_seeds_with_scores if score >= SIMILARITY_THRESHOLD]
    top_second_seeds = [chunk_id for chunk_id, _, _ in second_seeds_with_scores[:TOP_SEEDS_FOR_EXPANSION]]
    
    second_expanded_ids = []
    if top_second_seeds:
        second_expanded_ids = document_graph.expand_from_chunks(top_second_seeds, max_expansion=12)
    
    # Filter out already retrieved chunks
    new_expanded = [cid for cid in second_expanded_ids if cid not in existing_chunk_ids]
    
    all_second_ids = list(set(second_seed_ids + new_expanded))
    logger.info(f"Graph expansion: {len(new_expanded)} additional chunks (total new: {len(all_second_ids)})")
    
    # Get Document objects (no re-ranking)
    second_chunks = []
    chunk_dict = {chunk.metadata.get("chunk_index"): chunk for chunk in _chunks}
    for chunk_id in all_second_ids:
        if chunk_id in chunk_dict:
            second_chunks.append(chunk_dict[chunk_id])
    second_chunks = second_chunks[:15]  # Cap at 15

    state["second_seed_ids"] = second_seed_ids
    state["second_seed_scores"] = second_seed_scores
    state["second_expanded_ids"] = new_expanded
    state["second_retrieval_chunks"] = second_chunks
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    # Update debug info
    debug_info = state.get("debug_info", {})
    debug_info["second_retrieval"] = {
        "total_seed_chunks": len(second_seed_ids),
        "high_quality_seeds": len(high_quality_second_seeds),
        "seeds_used_for_expansion": len(top_second_seeds),
        "graph_expanded_chunks": len(new_expanded),
        "total_second_chunks": len(second_chunks),
    }
    state["debug_info"] = debug_info
    
    logger.info(f"Second retrieval complete: {len(second_chunks)} new chunks")
    return state

def generate_final_answer(state: AgentState) -> AgentState:
    """Generate final answer from all retrieved chunks (re-ranked and filtered)"""
    # Always use the user's original question; fallback to refined query if missing (e.g. state merge issue)
    query = (state.get("query") or "").strip() or (state.get("new_query") or "").strip()
    if not query:
        logger.warning("No question in state (query and new_query both empty); cannot generate answer.")
        state["final_answer"] = "Error: No question was provided. Please ask a question about the document."
        return state

    logger.info(f"Generating answer for question: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    # Use re-ranked chunks if available, otherwise use retrieved chunks
    primary_chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])
    second_chunks = state.get("second_retrieval_chunks", [])
    
    # Combine chunks: when second retrieval was used (refined query), put those chunks first
    # so the answer is based on the most relevant retrieval (e.g. "jurisdiction" refined query).
    primary_chunk_ids = {chunk.metadata.get("chunk_index") for chunk in primary_chunks}
    second_only = [c for c in second_chunks if c.metadata.get("chunk_index") not in primary_chunk_ids]
    if second_only:
        # Refined-query chunks first, then primary, so LLM prioritizes jurisdiction (etc.) content
        all_chunks = second_only + primary_chunks
        max_chunks = 35
    else:
        all_chunks = primary_chunks.copy()
        max_chunks = 25
    chunks_to_use = all_chunks[:max_chunks]
    
    llm = _llm
    
    logger.info(f"Generating final answer from {len(all_chunks)} chunks (primary: {len(primary_chunks)}, second: {len(second_chunks)})...")
    
    # Prepare comprehensive context
    chunks_text = []
    rerank_scores = state.get("rerank_scores", {})
    
    for i, chunk in enumerate(chunks_to_use, 1):
        chunk_id = chunk.metadata.get("chunk_index")
        heading = chunk.metadata.get("heading", "No heading")
        section = chunk.metadata.get("section_path", "No section")
        summary = chunk.metadata.get("summary", "")
        content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
        
        # Add relevance score if available
        relevance_note = ""
        if chunk_id in rerank_scores:
            relevance_note = f" (Relevance: {rerank_scores[chunk_id]:.2f})"
        
        chunks_text.append(f"""
[Chunk {i}]
Section: {section}
Heading: {heading}
Summary: {summary}
Content: {content}{relevance_note}
""")
    
    context = "\n".join(chunks_text)
    
    # Put the question first and repeat before answer so the model always sees it
    answer_prompt = f"""QUESTION (answer this using only the document chunks below):
{query}

---
Document chunks (use only this information):
{context}
---

Again, the question to answer is: {query}

Answer (use only the chunks above; no chunk numbers; markdown ok; answer in detail and descriptive way):"""
    
    try:
        response = llm.invoke(answer_prompt)
        response_str = (getattr(response, "content", None) or str(response)).strip()
        final_answer = response_str
        _append_token_usage(
            state, "generate_answer",
            input_tokens=_est_tokens(answer_prompt),
            output_tokens=_est_tokens(response_str),
        )
        
        # Post-process to remove any chunk references that might have slipped through
        # Remove patterns like "Chunk 12", "(Chunk 12)", "[Chunk 12]", etc.
        # IMPORTANT: Preserve newlines and markdown formatting!
        final_answer = re.sub(r'\([Cc]hunk\s+\d+\)', '', final_answer)
        final_answer = re.sub(r'\[[Cc]hunk\s+\d+\]', '', final_answer)
        final_answer = re.sub(r'[Cc]hunk\s+\d+', '', final_answer)
        
        # Clean up punctuation issues but preserve newlines
        # Only collapse multiple spaces on the same line (not newlines)
        final_answer = re.sub(r'[ \t]+', ' ', final_answer)  # Multiple spaces/tabs to single space
        final_answer = re.sub(r'\s*,\s*,', ',', final_answer)  # Double commas
        final_answer = re.sub(r'\s*\.\s*\.', '.', final_answer)  # Double periods
        # Clean up spaces before punctuation (but preserve newlines)
        final_answer = re.sub(r' +([,.!?;:])', r'\1', final_answer)
        # Remove trailing spaces from lines (but keep the newline)
        final_answer = re.sub(r' +(\n)', r'\1', final_answer)
        # Clean up multiple consecutive newlines (keep max 2 for paragraph breaks)
        final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)
        final_answer = final_answer.strip()
        
        # Validate that answer is based on retrieved chunks
        if not all_chunks:
            logger.warning("No chunks available for answer generation")
            state["final_answer"] = "I could not retrieve sufficient information from the document to answer this question."
        else:
            state["final_answer"] = final_answer
            logger.info(f"Final answer generated from {len(chunks_to_use)} chunks")
        
        # Update debug info
        debug_info = state.get("debug_info", {})
        debug_info["answer_generation"] = {
            "total_chunks_available": len(all_chunks),
            "chunks_used": len(chunks_to_use),
            "primary_chunks": len(primary_chunks),
            "second_chunks": len(second_chunks),
            "answer_length": len(final_answer) if all_chunks else 0,
        }
        state["debug_info"] = debug_info
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        state["final_answer"] = f"Error generating answer: {str(e)}"
        debug_info = state.get("debug_info", {})
        debug_info["answer_generation"] = {"error": str(e)}
        state["debug_info"] = debug_info
    
    return state

# ---------------- CONDITIONAL EDGE FUNCTIONS ---------------- 
def route_query_type(state: AgentState) -> Literal["summarize_page", "normal_retrieval"]:
    """Route query to page summarization or normal retrieval"""
    is_page_summary = state.get("is_page_summary", False)
    page_number = state.get("page_number")
    
    if is_page_summary and page_number:
        logger.info(f"Routing to page summarization for page {page_number}")
        return "summarize_page"
    else:
        logger.info("Routing to normal retrieval")
        return "normal_retrieval"

def should_continue_search(state: AgentState) -> Literal["second_retrieval", "generate_answer"]:
    """Decide whether to do second retrieval or generate answer"""
    needs_more = state.get("needs_more_info", False)
    iteration = state.get("iteration_count", 0)
    
    if iteration >= 3:
        logger.info("Maximum iterations reached, generating answer")
        return "generate_answer"
    
    if needs_more and state.get("new_query"):
        logger.info("Proceeding to second graph-enhanced retrieval")
        return "second_retrieval"
    else:
        logger.info("Sufficient information, generating answer")
        return "generate_answer"

# ---------------- WORKFLOW CREATION ---------------- 
def create_retrieval_agent(vector_store: Chroma, document_graph: DocumentGraph, chunks: List[Document], llm: Any, document_folder: Optional[Path] = None) -> StateGraph:
    """Create the LangGraph retrieval agent with page summarization support"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("summarize_page", summarize_page)
    workflow.add_node("initial_retrieval", initial_retrieval)
    workflow.add_node("analyze_chunks", analyze_chunks)
    workflow.add_node("second_retrieval", second_retrieval)
    workflow.add_node("generate_answer", generate_final_answer)
    
    # Define edges
    workflow.set_entry_point("classify_query")
    
    # Route based on query type
    workflow.add_conditional_edges(
        "classify_query",
        route_query_type,
        {
            "summarize_page": "summarize_page",
            "normal_retrieval": "initial_retrieval"
        }
    )
    
    # Page summarization path
    workflow.add_edge("summarize_page", END)
    
    # Normal retrieval path
    workflow.add_edge("initial_retrieval", "analyze_chunks")
    
    # Conditional edge for second retrieval
    workflow.add_conditional_edges(
        "analyze_chunks",
        should_continue_search,
        {
            "second_retrieval": "second_retrieval",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_edge("second_retrieval", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# ---------------- HELPER FUNCTIONS ---------------- 
def find_vector_mapping_file(directory: Path) -> Optional[Path]:
    """Find vector mapping JSON file in directory"""
    # First, check for unified_mapping.json (for unified/multi-PDF graphs)
    unified_file = directory / "unified_mapping.json"
    if unified_file.exists():
        return unified_file
    
    # Look for files matching pattern: *_vector_mapping.json
    pattern = "*_vector_mapping.json"
    mapping_files = list(directory.glob(pattern))
    
    if not mapping_files:
        # Try alternative pattern: file_vector_mapping.json
        alt_file = directory / "file_vector_mapping.json"
        if alt_file.exists():
            return alt_file
        return None
    
    # If multiple files found, use the most recent one
    if len(mapping_files) > 1:
        logger.info(f"Found {len(mapping_files)} mapping files, using most recent:")
        for mf in sorted(mapping_files, key=lambda p: p.stat().st_mtime, reverse=True):
            logger.info(f"  - {mf.name}")
    
    return max(mapping_files, key=lambda p: p.stat().st_mtime)

def find_graph_file(directory: Path) -> Optional[Path]:
    """Find document graph JSON file in directory"""
    # First, check for unified_graph.json (for unified/multi-PDF graphs)
    unified_file = directory / "unified_graph.json"
    if unified_file.exists():
        return unified_file
    
    # Look for files matching pattern: *_document_graph.json
    pattern = "*_document_graph.json"
    graph_files = list(directory.glob(pattern))
    
    if not graph_files:
        # Try alternative pattern: file_document_graph.json
        alt_file = directory / "file_document_graph.json"
        if alt_file.exists():
            return alt_file
        return None
    
    # If multiple files found, use the most recent one
    if len(graph_files) > 1:
        logger.info(f"Found {len(graph_files)} graph files, using most recent:")
        for gf in sorted(graph_files, key=lambda p: p.stat().st_mtime, reverse=True):
            logger.info(f"  - {gf.name}")
    
    return max(graph_files, key=lambda p: p.stat().st_mtime)

def find_vector_db_path(directory: Path, document_stem: Optional[str] = None) -> Optional[Path]:
    """Find vector database directory"""
    vector_db_dir = directory / "vector_db"
    
    if not vector_db_dir.exists():
        return None
    
    # First, check for unified vector store (for unified/multi-PDF graphs)
    unified_db = vector_db_dir / "unified"
    if unified_db.exists():
        return unified_db
    
    # If document_stem is provided, try that first
    if document_stem:
        db_path = vector_db_dir / document_stem
        if db_path.exists():
            return db_path
    
    # Otherwise, find any subdirectory in vector_db
    subdirs = [d for d in vector_db_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Use the most recently modified directory
    if len(subdirs) > 1:
        logger.info(f"Found {len(subdirs)} vector DB directories, using most recent:")
        for sd in sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True):
            logger.info(f"  - {sd.name}")
    
    return max(subdirs, key=lambda p: p.stat().st_mtime)

# ---------------- MAIN FUNCTION ---------------- 
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Graph-enhanced retrieval agent using LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use folder name (auto-detects E subfolder)
  python retrivalAgentE.py output/HDFC-Life-Cancer-Care-101N106V04-Policy-Document
  
  # Use most recent folder in output directory
  python retrivalAgentE.py
  
  # Specify explicit paths (overrides auto-detection)
  python retrivalAgentE.py output/folder --mapping-file path/to/mapping.json --graph-file path/to/graph.json
        """
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to folder containing document (will look for E subfolder), or use most recent folder if not specified"
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        help="Explicit path to vector mapping JSON file (overrides auto-detection)"
    )
    parser.add_argument(
        "--graph-file",
        type=str,
        help="Explicit path to document graph JSON file (overrides auto-detection)"
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        help="Explicit path to vector database directory (overrides auto-detection)"
    )
    
    args = parser.parse_args()
    
    # Determine base directory and files
    if args.path:
        input_path = Path(args.path)
    else:
        # Default: look for most recent folder in output directory
        output_dir = Path("output")
        if output_dir.exists():
            folders = [d for d in output_dir.iterdir() if d.is_dir()]
            if folders:
                input_path = max(folders, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent folder: {input_path}")
            else:
                logger.error("No folders found in output directory. Please specify a folder path.")
                sys.exit(1)
        else:
            logger.error("Output directory not found. Please specify a folder path.")
            sys.exit(1)
    
    # If input_path is a file, use its parent directory
    if input_path.is_file():
        base_folder = input_path.parent
    elif input_path.is_dir():
        base_folder = input_path
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)
    
    # Look for E subfolder (Plan E output)
    plan_e_dir = base_folder / "E"
    
    # Auto-detect or use explicit paths
    if args.mapping_file:
        vector_mapping_file = Path(args.mapping_file)
        document_stem = vector_mapping_file.stem.replace("_vector_mapping", "")
    else:
        if plan_e_dir.exists():
            vector_mapping_file = find_vector_mapping_file(plan_e_dir)
        else:
            vector_mapping_file = find_vector_mapping_file(base_folder)
        if vector_mapping_file:
            document_stem = vector_mapping_file.stem.replace("_vector_mapping", "")
        else:
            document_stem = None
    
    # Find graph file
    if args.graph_file:
        graph_file = Path(args.graph_file)
    else:
        if plan_e_dir.exists():
            graph_file = find_graph_file(plan_e_dir)
        else:
            graph_file = find_graph_file(base_folder)
    
    # Find vector DB path
    if args.vector_db:
        vector_db_path = Path(args.vector_db)
    else:
        if plan_e_dir.exists():
            vector_db_path = find_vector_db_path(plan_e_dir, document_stem)
        else:
            vector_db_path = find_vector_db_path(base_folder, document_stem)
    
    # Validate files
    if not vector_mapping_file or not vector_mapping_file.exists():
        logger.error(f"Vector mapping file not found in: {base_folder}")
        logger.error("Please run vectorizerE.py first to generate the vector mapping")
        logger.error("Or specify the file with --mapping-file")
        sys.exit(1)
    
    if not graph_file or not graph_file.exists():
        logger.error(f"Graph file not found in: {base_folder}")
        logger.error("Please run vectorizerE.py first to generate the graph")
        logger.error("Or specify the file with --graph-file")
        sys.exit(1)
    
    if not vector_db_path or not vector_db_path.exists():
        logger.error(f"Vector DB path not found")
        logger.error("Please run vectorizerE.py first to generate the vector database")
        logger.error("Or specify the path with --vector-db")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("RetrievalAgentE - Plan E: Graph-Enhanced Retrieval Agent")
    logger.info("=" * 80)
    logger.info(f"Document folder: {base_folder}")
    logger.info(f"Vector mapping: {vector_mapping_file.name}")
    logger.info(f"Graph file: {graph_file.name}")
    logger.info(f"Vector DB: {vector_db_path.name}")
    
    # Load data
    chunks = load_chunks_from_mapping(vector_mapping_file)
    if not chunks:
        logger.error("Failed to load chunks. Exiting.")
        sys.exit(1)
    
    document_graph = DocumentGraph()
    document_graph.load(graph_file)
    
    vector_store = load_vector_store(vector_db_path)
    llm = get_llm(temperature=0.3)
    
    logger.info("=" * 80)
    logger.info("Data loaded successfully!")
    logger.info(f"Graph: {len(document_graph.graph.nodes)} nodes, {len(document_graph.graph.edges)} edges")
    logger.info("=" * 80)
    
    # Set global resources (include document folder for page summarization)
    set_agent_resources(vector_store, document_graph, chunks, llm, base_folder)
    
    # Create agent (include document folder for page summarization)
    agent = create_retrieval_agent(vector_store, document_graph, chunks, llm, base_folder)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("GRAPH-ENHANCED RETRIEVAL AGENT - Interactive Mode")
    print("=" * 80)
    print("Enter queries. The agent will:")
    print("  1. Perform vector similarity search (seed chunks)")
    print("  2. Expand via graph traversal (sections, adjacent chunks)")
    print("  3. Analyze if more information is needed")
    print("  4. Optionally perform a second graph-enhanced search")
    print("  5. Generate a comprehensive answer")
    print("Type 'exit' or 'quit' to exit")
    print("=" * 80)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if not query or query.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            # Initialize state
            initial_state: AgentState = {
                "query": query,
                "is_page_summary": False,
                "page_number": None,
                "seed_chunk_ids": [],
                "graph_expanded_ids": [],
                "retrieved_chunks": [],
                "chunk_analysis": "",
                "needs_more_info": False,
                "new_query": None,
                "second_seed_ids": [],
                "second_expanded_ids": [],
                "second_retrieval_chunks": [],
                "final_answer": "",
                "iteration_count": 0,
                "document_folder": None,
            }
            
            # Run agent
            print("\n" + "=" * 80)
            print("AGENT EXECUTION")
            print("=" * 80)
            
            final_state = agent.invoke(initial_state)
            
            # Display results
            print("\n" + "=" * 80)
            print("FINAL ANSWER")
            print("=" * 80)
            print(final_state["final_answer"])
            print("=" * 80)
            
            if final_state.get("chunk_analysis"):
                print("\n" + "=" * 80)
                print("CHUNK ANALYSIS")
                print("=" * 80)
                print(final_state["chunk_analysis"])
                print("=" * 80)
            
            print(f"\nRetrieval Statistics:")
            print(f"  - Seed chunks (vector search): {len(final_state['seed_chunk_ids'])}")
            print(f"  - Graph-expanded chunks: {len(final_state['graph_expanded_ids'])}")
            print(f"  - Total initial chunks: {len(final_state['retrieved_chunks'])}")
            if final_state.get("second_seed_ids"):
                print(f"  - Second seed chunks: {len(final_state['second_seed_ids'])}")
                print(f"  - Second expanded chunks: {len(final_state['second_expanded_ids'])}")
                print(f"  - Total second chunks: {len(final_state['second_retrieval_chunks'])}")
            print(f"  - Total chunks used: {len(final_state['retrieved_chunks']) + len(final_state.get('second_retrieval_chunks', []))}")
            
            if final_state.get("new_query"):
                print(f"  - Second query used: {final_state['new_query']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
