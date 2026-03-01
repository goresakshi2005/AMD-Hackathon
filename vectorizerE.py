"""
VectorizerE - Plan E Implementation: Graph-Based Document Structure
Converts markdown documents to vector embeddings with knowledge graph structure.
Uses graph traversal for enhanced retrieval that captures document relationships.
Uses inference_config for embeddings and LLM (Ollama or Hugging Face).
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import defaultdict
from dataclasses import dataclass, field

# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available. Install with: pip install networkx")

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_community.vectorstores.utils import filter_complex_metadata

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

# Inference provider (Ollama or Hugging Face)
from config.inference_config import (
    get_embeddings,
    get_llm,
    check_inference_ready,
    get_embedding_model_id,
    get_llm_model_id,
)

# Token limits
EMBEDDING_MAX_TOKENS = 1500
LLM_MAX_TOKENS = 8192
LLM_OPTIMAL_TOKENS = 4000

# Character limits
EMBEDDING_MAX_CHARS = 2000
CHUNK_SIZE = EMBEDDING_MAX_CHARS
CHUNK_OVERLAP = 150

# Token counting encoding
TOKEN_ENCODING = "cl100k_base" if TIKTOKEN_AVAILABLE else None

# ---------------- TOKEN TRACKING ---------------- 
class TokenTracker:
    """Track token usage for different models"""
    
    def __init__(self):
        self.encoding = None
        if TIKTOKEN_AVAILABLE and TOKEN_ENCODING:
            try:
                self.encoding = tiktoken.get_encoding(TOKEN_ENCODING)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
        
        self.stats = {
            "embedding_tokens": 0,
            "llm_tokens": 0,
            "total_chunks": 0,
            "truncated_chunks": 0
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken (approximate for WordPiece models)"""
        if not self.encoding:
            return int(len(text) / 1.5)
        
        try:
            tiktoken_count = len(self.encoding.encode(text))
            wordpiece_estimate = int(tiktoken_count * 1.3)
            return wordpiece_estimate
        except Exception as e:
            logger.warning(f"Token counting error: {e}, using fallback")
            return int(len(text) / 1.5)
    
    def check_embedding_limit(self, text: str) -> Tuple[str, bool]:
        """Check if text exceeds embedding limit, truncate if needed"""
        if len(text) > EMBEDDING_MAX_CHARS:
            self.stats["truncated_chunks"] += 1
            truncated = text[:EMBEDDING_MAX_CHARS]
            last_period = truncated.rfind('. ')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > len(truncated) * 0.7:
                truncated = truncated[:cut_point + 1] + "..."
            else:
                truncated = truncated + "..."
            
            tokens = self.count_tokens(truncated)
            self.stats["embedding_tokens"] += tokens
            if tokens > EMBEDDING_MAX_TOKENS:
                logger.warning(f"After char truncation, still {tokens} tokens")
                truncated = self._truncate_to_tokens(truncated, int(EMBEDDING_MAX_TOKENS * 0.9))
            return truncated, True
        
        tokens = self.count_tokens(text)
        SAFETY_MARGIN = int(EMBEDDING_MAX_TOKENS * 0.85)
        
        if tokens > SAFETY_MARGIN:
            self.stats["truncated_chunks"] += 1
            truncated = self._truncate_to_tokens(text, SAFETY_MARGIN)
            new_tokens = self.count_tokens(truncated)
            self.stats["embedding_tokens"] += new_tokens
            if new_tokens > EMBEDDING_MAX_TOKENS:
                logger.warning(f"Truncation may not be sufficient: {new_tokens} tokens")
            return truncated, True
        
        self.stats["embedding_tokens"] += tokens
        return text, False
    
    def check_llm_limit(self, text: str, max_tokens: int = LLM_OPTIMAL_TOKENS) -> Tuple[str, bool]:
        """Check if text exceeds LLM limit"""
        tokens = self.count_tokens(text)
        self.stats["llm_tokens"] += tokens
        
        if tokens > max_tokens:
            truncated = self._truncate_to_tokens(text, max_tokens)
            return truncated, True
        
        return text, False
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not self.encoding:
            max_chars = int(max_tokens * 1.5)
            if len(text) <= max_chars:
                return text
            truncated = text[:max_chars]
            last_period = truncated.rfind('. ')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > max_chars * 0.7:
                return truncated[:cut_point + 1] + "..."
            return truncated + "..."
        
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            safe_max = int(max_tokens * 0.95)
            truncated_tokens = tokens[:safe_max]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            verify_tokens = self.encoding.encode(truncated_text)
            if len(verify_tokens) > max_tokens:
                truncated_tokens = tokens[:int(max_tokens * 0.9)]
                truncated_text = self.encoding.decode(truncated_tokens)
            
            last_period = truncated_text.rfind('. ')
            last_newline = truncated_text.rfind('\n')
            cut_point = max(last_period, last_newline)
            
            if cut_point > len(truncated_text) * 0.7:
                final_text = truncated_text[:cut_point + 1] + "..."
            else:
                final_text = truncated_text + "..."
            
            final_tokens = self.encoding.encode(final_text)
            if len(final_tokens) > max_tokens:
                hard_truncated = tokens[:int(max_tokens * 0.9)]
                final_text = self.encoding.decode(hard_truncated) + "..."
            
            return final_text
        except Exception as e:
            logger.warning(f"Token truncation error: {e}, using character truncation")
            max_chars = int(max_tokens * 1.5)
            return text[:max_chars] + "..."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset stats for a new vectorization run (so each document is counted separately)."""
        self.stats = {
            "embedding_tokens": 0,
            "llm_tokens": 0,
            "total_chunks": 0,
            "truncated_chunks": 0,
        }

# Global token tracker
token_tracker = TokenTracker()

# ---------------- DOCUMENT GRAPH ---------------- 
class DocumentGraph:
    """Knowledge graph for document structure"""
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for Plan E. Install with: pip install networkx")
        
        self.graph = nx.DiGraph()  # Directed graph
        self.chunk_nodes = {}  # chunk_id -> node_id
        self.section_nodes = {}  # section_path -> node_id
        self.page_nodes = {}  # page_number -> node_id
    
    def add_section_node(self, section_path: str, section_title: str, level: int, start_line: int):
        """Add a section node to the graph"""
        node_id = f"section:{section_path}"
        if node_id not in self.graph:
            self.graph.add_node(node_id, 
                              type="section",
                              section_path=section_path,
                              section_title=section_title,
                              level=level,
                              start_line=start_line)
            self.section_nodes[section_path] = node_id
        return node_id
    
    def add_chunk_node(self, chunk_id: int, chunk: Document):
        """Add a chunk node to the graph"""
        node_id = f"chunk:{chunk_id}"
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                              type="chunk",
                              chunk_id=chunk_id,
                              heading=chunk.metadata.get("heading", ""),
                              section_path=chunk.metadata.get("section_path", ""),
                              page_number=chunk.metadata.get("page_number"))
            self.chunk_nodes[chunk_id] = node_id
        return node_id
    
    def add_page_node(self, page_number: int, classification: Optional[str] = None):
        """Add a page node to the graph"""
        if page_number <= 0:
            return None
        
        node_id = f"page:{page_number}"
        if node_id not in self.graph:
            node_data = {
                "type": "page",
                "page_number": page_number
            }
            if classification:
                node_data["classification"] = classification
            self.graph.add_node(node_id, **node_data)
            self.page_nodes[page_number] = node_id
        else:
            # Update existing node with classification if provided
            if classification:
                self.graph.nodes[node_id]["classification"] = classification
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, relation: str, **kwargs):
        """Add an edge between nodes"""
        if source_id and target_id:
            self.graph.add_edge(source_id, target_id, relation=relation, **kwargs)
    
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
        
        return adjacent_ids[:window * 2]  # window chunks before + window chunks after
    
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
    
    def expand_from_chunks(self, chunk_ids: List[int], max_expansion: int = 20) -> List[int]:
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
    
    def save(self, filepath: Path):
        """Save graph to JSON format"""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node_id,
                **node_data
            })
        
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                **edge_data
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    def load(self, filepath: Path):
        """Load graph from JSON format"""
        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
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
        
        logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

# ---------------- DOCUMENT STRUCTURE EXTRACTION ---------------- 
def extract_document_structure(markdown_content: str) -> Dict[str, Any]:
    """Extract document structure: sections, headers, tables"""
    lines = markdown_content.split("\n")
    
    structure = {
        "sections": [],
        "headers": [],
        "tables": []
    }
    
    current_section_path = []  # List of (level, title) tuples to maintain hierarchy
    line_number = 0
    in_table = False
    table_start = None
    
    for line in lines:
        line_number += 1
        
        # Detect tables
        if line.strip().startswith("|") and "|" in line:
            if not in_table:
                in_table = True
                table_start = line_number
        else:
            if in_table:
                structure["tables"].append({
                    "start_line": table_start,
                    "end_line": line_number - 1,
                    "section": current_section_path[-1][1] if current_section_path else None
                })
                in_table = False
        
        # Detect headers
        header_match = re.match(r'^(#+)\s+(.+)$', line.strip())
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            
            # Build hierarchical section path - keep full hierarchy
            # Remove any headings at the same or deeper level
            # Keep only headings at shallower levels (lower level numbers)
            current_section_path = [(l, t) for l, t in current_section_path if l < level]
            
            # Build full hierarchical path from all parent sections
            # Example: if we have # Part A, ## Part B, ### Section 1
            # For ### Section 1, path will be "Part A > Part B > Section 1" (full hierarchy)
            if current_section_path:
                # Build path from all parent sections
                parent_titles = [t for l, t in current_section_path]
                section_path_str = " > ".join(parent_titles + [title])
            else:
                # No parent sections, just use the title
                section_path_str = title
            
            # Update the current section path for tracking (keep full hierarchy)
            current_section_path.append((level, title))
            
            header_info = {
                "level": level,
                "title": title,
                "line": line_number,
                "section_path": section_path_str
            }
            structure["headers"].append(header_info)
            
            current_section = {
                "path": section_path_str,
                "title": title,
                "level": level,
                "start_line": line_number
            }
            structure["sections"].append(current_section)
    
    return structure

def get_section_for_line(line_number: int, structure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the most specific (deepest) section that contains a given line number.
    
    This finds the section with the highest level (most nested) that starts before or at the line number.
    This ensures chunks are assigned to the most specific section they belong to.
    """
    sections = structure["sections"]
    
    # Find all sections that contain this line (start_line <= line_number)
    containing_sections = [s for s in sections if s["start_line"] <= line_number]
    
    if not containing_sections:
        return None
    
    # Find the section with the highest level (most nested/deepest)
    # If levels are equal, prefer the one that starts closest to the line (most recent)
    most_specific = max(containing_sections, key=lambda s: (s.get("level", 0), s["start_line"]))
    
    return most_specific

# ---------------- ENHANCED CHUNKING ---------------- 
def is_chunk_empty(chunk: Document) -> bool:
    """Check if a chunk is empty or has minimal content (only headers, whitespace, etc.)"""
    content = chunk.page_content.strip()
    
    # Empty or only whitespace
    if not content or len(content) < 10:
        return True
    
    # Count non-header, non-whitespace lines
    lines = content.split("\n")
    meaningful_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, headers, and horizontal rules
        if stripped and not stripped.startswith("#") and not re.match(r'^[-=*]{3,}$', stripped):
            meaningful_lines.append(stripped)
    
    # If we have less than 2 meaningful lines or very short content, consider it empty
    if len(meaningful_lines) < 2:
        return True
    
    # Check if meaningful content is too short (less than 50 chars of actual text)
    meaningful_text = " ".join(meaningful_lines)
    if len(meaningful_text) < 50:
        return True
    
    return False

def parse_markdown_enhanced(markdown_content: str, page_mapping: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], Dict[str, Any]]:
    """Parse markdown with structure awareness and enhanced metadata"""
    logger.info("Extracting document structure...")
    structure = extract_document_structure(markdown_content)
    
    # Log heading level distribution to verify all levels are captured
    level_counts = {}
    for header in structure['headers']:
        level = header.get('level', 0)
        level_counts[level] = level_counts.get(level, 0) + 1
    
    logger.info(f"Found {len(structure['sections'])} sections, {len(structure['headers'])} headers, {len(structure['tables'])} tables")
    if level_counts:
        level_summary = ", ".join([f"Level {k}: {v}" for k, v in sorted(level_counts.items())])
        logger.info(f"Heading level distribution: {level_summary}")
    
    # Load page mapping if provided
    line_to_page = None
    if page_mapping:
        line_to_page = page_mapping.get("line_to_page", [])
        if line_to_page:
            logger.info(f"Using page mapping: {len(line_to_page)} lines mapped to pages")
        else:
            logger.warning("Page mapping provided but line_to_page is empty")
    
    lines = markdown_content.split("\n")
    chunks = []
    current_chunk_lines = []
    current_chunk_size = 0
    chunk_index = 0
    current_section = None
    current_section_path = ""
    
    for line_num, line in enumerate(lines, 1):
        # Update current section as we process lines
        section = get_section_for_line(line_num, structure)
        if section and section != current_section:
            current_section = section
            current_section_path = section["path"]
        
        is_header = bool(re.match(r'^#+\s+', line.strip()))
        
        line_with_newline = line + "\n"
        line_size = len(line_with_newline)
        
        if current_chunk_size + line_size > CHUNK_SIZE and current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            chunk_start_line = line_num - len(current_chunk_lines)
            
            # Get the section for the CHUNK'S START LINE (not current line)
            # This ensures chunks are assigned to the correct section
            chunk_section = get_section_for_line(chunk_start_line, structure)
            chunk_section_path = chunk_section["path"] if chunk_section else ""
            
            # Determine page number for this chunk (use start line's page)
            page_number = None
            if line_to_page and chunk_start_line > 0 and chunk_start_line <= len(line_to_page):
                page_number = line_to_page[chunk_start_line - 1]  # Convert to 0-based index
            chunk = create_enhanced_chunk(
                chunk_text,
                chunk_index,
                chunk_section_path,
                chunk_section,
                structure,
                chunk_start_line,
                page_number
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Calculate minimal overlap: use only 1-2 lines for context
            # Overlap should be minimal to avoid duplicate content
            # Use maximum 2 lines or ~5% of chunk, whichever is smaller
            overlap_size = min(max(1, len(current_chunk_lines) // 20), 2)
            overlap_lines = current_chunk_lines[-overlap_size:] if len(current_chunk_lines) > overlap_size else []
            
            # Reset for next chunk with minimal overlap
            current_chunk_lines = overlap_lines
            current_chunk_size = sum(len(l) for l in current_chunk_lines)
        
        current_chunk_lines.append(line_with_newline)
        current_chunk_size += line_size
    
    if current_chunk_lines:
        chunk_text = "".join(current_chunk_lines)
        chunk_start_line = len(lines) - len(current_chunk_lines)
        
        # Get the section for the CHUNK'S START LINE (not current line)
        # This ensures chunks are assigned to the correct section
        chunk_section = get_section_for_line(chunk_start_line, structure)
        chunk_section_path = chunk_section["path"] if chunk_section else ""
        
        # Determine page number for this chunk
        page_number = None
        if line_to_page and chunk_start_line > 0 and chunk_start_line <= len(line_to_page):
            page_number = line_to_page[chunk_start_line - 1]  # Convert to 0-based index
        chunk = create_enhanced_chunk(
            chunk_text,
            chunk_index,
            chunk_section_path,
            chunk_section,
            structure,
            chunk_start_line,
            page_number
        )
        chunks.append(chunk)
    
    # Filter out empty chunks and duplicate chunks before adding adjacency metadata
    filtered_chunks = []
    seen_content_hashes = set()  # Track content to avoid duplicates
    
    for chunk in chunks:
        if is_chunk_empty(chunk):
            logger.debug(f"Skipping empty chunk: {chunk.metadata.get('heading', 'Unknown')}")
            continue
        
        # Check for duplicate/similar content (90%+ overlap)
        # Normalize: remove extra whitespace for comparison
        content_normalized = " ".join(chunk.page_content.split())
        
        # Check against all previously seen chunks for high similarity
        is_duplicate = False
        for seen_content in seen_content_hashes:
            # Compare first 1500 chars (catches most of chunk content)
            current_preview = content_normalized[:1500]
            seen_preview = seen_content[:1500]
            
            # Calculate word-based similarity (Jaccard similarity)
            current_words = set(current_preview.split())
            seen_words = set(seen_preview.split())
            
            if len(current_words) > 0 and len(seen_words) > 0:
                # Jaccard similarity: intersection over union
                intersection = len(current_words & seen_words)
                union = len(current_words | seen_words)
                similarity = intersection / union if union > 0 else 0.0
                
                # If 90%+ similar, consider it a duplicate
                if similarity >= 0.90:
                    logger.debug(f"Skipping duplicate chunk (similarity: {similarity:.1%}): {chunk.metadata.get('heading', 'Unknown')} (start_line: {chunk.metadata.get('start_line')})")
                    is_duplicate = True
                    break
        
        if is_duplicate:
            continue
        
        # Store normalized content for future comparisons
        seen_content_hashes.add(content_normalized)
        filtered_chunks.append(chunk)
    
    # Update chunk indices after filtering
    for i, chunk in enumerate(filtered_chunks):
        chunk.metadata["chunk_index"] = i
    
    # Add adjacency metadata
    for i, chunk in enumerate(filtered_chunks):
        chunk.metadata["prev_chunk_ids"] = [i-2, i-1] if i >= 2 else [i-1] if i >= 1 else []
        chunk.metadata["next_chunk_ids"] = [i+1, i+2] if i < len(filtered_chunks)-2 else [i+1] if i < len(filtered_chunks)-1 else []
        chunk.metadata["sequential_position"] = i
    
    logger.info(f"Created {len(filtered_chunks)} chunks (filtered {len(chunks) - len(filtered_chunks)} empty chunks)")
    return filtered_chunks, structure

def create_enhanced_chunk(
    text: str,
    chunk_index: int,
    section_path: str,
    section: Optional[Dict[str, Any]],
    structure: Dict[str, Any],
    start_line: int,
    page_number: Optional[int] = None
) -> Document:
    """Create a chunk with enhanced metadata"""
    
    heading = ""
    lines = text.split("\n")
    
    # Strategy 1: If we have section information, prefer the section's title
    # This ensures chunks get the correct heading from their assigned section
    if section:
        section_title = section.get("title", "")
        if section_title:
            heading = section_title
    
    # Strategy 2: If no section title, try to find a header in the chunk content
    # Look through the entire chunk for any markdown header (all levels: #, ##, ###, ####, etc.)
    if not heading:
        for line in lines:
            line_stripped = line.strip()
            # Match any markdown header: #, ##, ###, ####, etc.
            header_match = re.match(r'^(#+)\s+(.+)$', line_stripped)
            if header_match:
                heading = header_match.group(2).strip()
                break  # Use the first header found
    
    # Strategy 3: If still no heading, use the last part of section_path (most specific heading)
    if not heading and section_path:
        path_parts = section_path.split(" > ")
        heading = path_parts[-1] if path_parts else ""
    
    # Strategy 4: Only as last resort, use generic chunk name
    if not heading:
        heading = f"Chunk {chunk_index + 1}"
    
    has_table = any("|" in line and line.strip().startswith("|") for line in lines)
    
    table_context = None
    if has_table:
        table_lines = [line for line in lines if "|" in line and line.strip().startswith("|")]
        if table_lines:
            table_context = "\n".join(table_lines[:3])
    
    metadata = {
        "chunk_index": chunk_index,
        "chunk_size": len(text),
        "heading": heading,
        "section_path": section_path or "",
        "section_title": section["title"] if section else "",
        "section_level": section["level"] if section else 0,
        "chunk_type": "table" if has_table else "text",
        "table_context": table_context,
        "start_line": start_line,
        "has_table": has_table,
        "page_number": page_number,
    }
    
    return Document(page_content=text, metadata=metadata)

# ---------------- LLM INTEGRATION ---------------- 
def classify_page_with_llm(page_content: str, llm: Any) -> str:
    """Classify a page using LLM based on its content - LLM generates its own label"""
    # Truncate to reasonable size for classification
    text_for_classification, _ = token_tracker.check_llm_limit(page_content[:3000], max_tokens=1000)
    
    prompt = f"""You are analyzing a single page of a document.

Your task:
- Read and understand the page content
- Determine what this page is primarily about (its main purpose or topic)
- Generate a short, descriptive label (1-4 words) that accurately describes this page

Return ONLY the label. No explanations, no prefixes, no additional text. Just the label itself.

Page content:
{text_for_classification}"""
    
    try:
        response = llm.invoke(prompt)
        response_text = (getattr(response, "content", None) or str(response)).strip()
        classification = response_text
        
        # Clean up the response
        classification = classification.split("\n")[0].strip()
        classification = re.sub(r'^[:\-\"\']+\s*', '', classification)
        classification = re.sub(r'\s*[:\-\"\']+$', '', classification)
        
        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            "label:",
            "classification:",
            "category:",
            "page type:",
            "this page is:",
            "the page is:"
        ]
        for prefix in prefixes_to_remove:
            if classification.lower().startswith(prefix):
                classification = classification[len(prefix):].strip()
                classification = re.sub(r'^[:\-\"\']+\s*', '', classification)
        
        # Ensure we have a valid label
        if not classification or len(classification) < 2:
            return "Unknown"
        
        # Limit length to reasonable size (max 50 chars)
        if len(classification) > 50:
            classification = classification[:47] + "..."
        
        return classification
    except Exception as e:
        logger.warning(f"Error classifying page: {e}")
        return "Unknown"

def classify_pages(chunks: List[Document], page_mapping: Optional[Dict[str, Any]], llm: Any) -> Dict[int, str]:
    """Classify all pages based on their chunk content"""
    if not page_mapping:
        logger.warning("No page mapping available, skipping page classification")
        return {}
    
    logger.info("Classifying pages based on content...")
    
    # Group chunks by page
    pages_content = {}  # page_number -> list of chunk contents
    for chunk in chunks:
        page_num = chunk.metadata.get("page_number")
        if page_num and page_num > 0:
            if page_num not in pages_content:
                pages_content[page_num] = []
            pages_content[page_num].append(chunk.page_content)
    
    # Classify each page
    page_classifications = {}
    total_pages = len(pages_content)
    
    for page_num, contents in sorted(pages_content.items()):
        if (page_num - 1) % 10 == 0 or page_num == 1:
            logger.info(f"Classifying page {page_num}/{total_pages}...")
        
        # Combine all chunks on this page
        page_text = "\n\n".join(contents)
        
        # Classify using LLM
        classification = classify_page_with_llm(page_text, llm)
        page_classifications[page_num] = classification
        
        logger.debug(f"Page {page_num}: {classification}")
    
    logger.info(f"Classified {len(page_classifications)} pages")
    return page_classifications

# ---------------- STATE DEFINITION ---------------- 
class VectorizerState(TypedDict):
    """State for the vectorization workflow"""
    markdown_file: str
    chunks: List[Document]
    structure: Dict[str, Any]
    processed_chunks: List[Dict[str, Any]]
    vector_store: Any
    document_graph: DocumentGraph
    json_mapping: List[Dict[str, Any]]
    page_mapping: Optional[Dict[str, Any]]
    page_classifications: Optional[Dict[int, str]]
    output_folder: Optional[str]
    token_usage: Optional[Dict[str, Any]]  # For economics tracker: embedding_tokens, llm_tokens, total_chunks, truncated_chunks

# ---------------- WORKFLOW NODES ---------------- 
def load_markdown(state: VectorizerState) -> VectorizerState:
    """Load and parse markdown file"""
    # md_path can be either a full path or a folder name
    md_path_or_folder = Path(state["markdown_file"])
    
    # If it's a folder, look for .md file inside it
    if md_path_or_folder.is_dir():
        md_folder = md_path_or_folder
        # Find .md file in the folder
        md_files = list(md_folder.glob("*.md"))
        if not md_files:
            logger.error(f"No .md file found in folder: {md_folder}")
            raise FileNotFoundError(f"No .md file found in {md_folder}")
        md_path = md_files[0]  # Use first .md file found
        logger.info(f"Found markdown file: {md_path.name} in folder: {md_folder}")
    else:
        # It's a file path
        md_path = md_path_or_folder
        md_folder = md_path.parent
    
    logger.info(f"Loading markdown file: {md_path}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Try to load page mapping if it exists (in same folder)
    page_mapping = None
    page_mapping_path = md_folder / f"{md_path.stem}_page_mapping.json"
    if page_mapping_path.exists():
        logger.info(f"Loading page mapping from: {page_mapping_path}")
        try:
            with open(page_mapping_path, 'r', encoding='utf-8') as f:
                page_mapping = json.load(f)
            logger.info(f"Loaded page mapping: {page_mapping.get('total_pages', 0)} pages")
        except Exception as e:
            logger.warning(f"Failed to load page mapping: {e}")
    else:
        logger.info(f"Page mapping not found at {page_mapping_path}, proceeding without page numbers")
    
    chunks, structure = parse_markdown_enhanced(markdown_content, page_mapping)
    
    state["chunks"] = chunks
    state["structure"] = structure
    state["page_mapping"] = page_mapping  # Store for later use in classification
    state["output_folder"] = str(md_folder)  # Store output folder for later use
    logger.info(f"Loaded and chunked {len(chunks)} chunks with structure awareness")
    return state

def process_chunks_one_by_one(state: VectorizerState) -> VectorizerState:
    """Process chunks: summarize, embed, and build graph"""
    chunks = state["chunks"]
    structure = state["structure"]
    output_folder = Path(state.get("output_folder", "output"))
    
    # Reset token stats so this run's usage is logged correctly (economics tracker)
    token_tracker.reset_stats()
    
    # Get document name from folder or markdown file
    if output_folder.is_dir():
        # Find .md file to get stem
        md_files = list(output_folder.glob("*.md"))
        if md_files:
            doc_stem = md_files[0].stem
        else:
            doc_stem = output_folder.name
    else:
        doc_stem = output_folder.stem
    
    # Create output directory structure: {output_folder}/E/
    plan_e_dir = output_folder / "E"
    plan_e_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = plan_e_dir / "vector_db" / doc_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = plan_e_dir / f"{doc_stem}_vector_mapping.json"
    graph_output_path = plan_e_dir / f"{doc_stem}_document_graph.json"
    
    # Initialize LLM and embeddings via inference_config (Ollama or Hugging Face)
    logger.info("Initializing LLM and embeddings...")
    llm = get_llm(temperature=0.3)
    embeddings = get_embeddings()
    
    # Initialize vector store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(output_dir)
    )
    
    # Initialize document graph
    document_graph = DocumentGraph()
    
    # Build graph structure - add ALL sections as nodes (not just those with chunks)
    # This ensures every heading becomes a node in the graph, preserving full hierarchy
    logger.info("Building document graph...")
    
    # Track which sections have chunks (for logging)
    sections_with_chunks = set()
    for chunk in chunks:
        section_path = chunk.metadata.get("section_path", "")
        if section_path:
            sections_with_chunks.add(section_path)
    
    # Build a map of all section paths to section info
    all_sections_map = {}
    for section in structure["sections"]:
        all_sections_map[section["path"]] = section
    
    # Add ALL sections from the structure as nodes (every heading should be a node)
    # This ensures complete hierarchical representation
    sections_to_add = set()
    for section in structure["sections"]:
        section_path = section["path"]
        sections_to_add.add(section_path)
        
        # Also add all parent sections in the hierarchy path
        if " > " in section_path:
            path_parts = section_path.split(" > ")
            current_path = ""
            for i, part in enumerate(path_parts):
                if i == 0:
                    current_path = part
                else:
                    current_path = current_path + " > " + part
                sections_to_add.add(current_path)
    
    # Add all section nodes (every heading becomes a node)
    for section_path in sorted(sections_to_add):
        if section_path in all_sections_map:
            section = all_sections_map[section_path]
            section_node = document_graph.add_section_node(
                section["path"],
                section["title"],
                section["level"],
                section["start_line"]
            )
        else:
            # Parent section not in structure (shouldn't happen, but handle gracefully)
            # This can happen if a parent section exists in the hierarchy but wasn't parsed
            # Extract the title from the path
            path_parts = section_path.split(" > ")
            title = path_parts[-1]
            
            # Determine level based on position in hierarchy (first part = level 2, second = level 3, etc.)
            # Level 1 = #, Level 2 = ##, Level 3 = ###, etc.
            level = len(path_parts) + 1  # First part is level 2 (##)
            
            # Try to find level from a similar section in structure
            for s in structure["sections"]:
                if s["path"] == section_path or s["title"] == title:
                    level = s["level"]
                    break
            
            # Try to find start_line from structure
            start_line = 0
            for s in structure["sections"]:
                if s["title"] == title:
                    start_line = s["start_line"]
                    break
            
            section_node = document_graph.add_section_node(
                section_path,
                title,
                level,
                start_line
            )
    
    # Create parent-child edges between sections
    logger.info("Creating parent-child relationships between sections...")
    section_edges_added = 0
    for section_path in sections_to_add:
        # Check if this section has a parent
        if " > " in section_path:
            # Extract parent path (everything except the last part)
            path_parts = section_path.split(" > ")
            parent_path = " > ".join(path_parts[:-1])
            
            # If parent section exists, create edge
            if parent_path in document_graph.section_nodes:
                child_node = document_graph.section_nodes.get(section_path)
                parent_node = document_graph.section_nodes.get(parent_path)
                if child_node and parent_node:
                    document_graph.add_edge(parent_node, child_node, relation="contains_section")
                    section_edges_added += 1
    
    logger.info(f"Created {len(sections_to_add)} section nodes (all headings from document structure)")
    logger.info(f"  - Sections with chunks: {len(sections_with_chunks)}")
    logger.info(f"  - Total sections in document: {len(structure['sections'])}")
    logger.info(f"Created {section_edges_added} parent-child section edges")
    
    # Classify pages if page mapping is available
    page_mapping = state.get("page_mapping")
    page_classifications = {}
    if page_mapping:
        page_classifications = classify_pages(chunks, page_mapping, llm)
        state["page_classifications"] = page_classifications
        logger.info(f"Page classification complete: {len(page_classifications)} pages classified")
    
    # Process chunks
    json_mapping = []
    processed_chunks = []
    
    logger.info(f"Processing {len(chunks)} chunks with graph structure...")
    
    for i, chunk in enumerate(chunks):
        # Skip empty chunks (double-check, though they should already be filtered)
        if is_chunk_empty(chunk):
            logger.debug(f"Skipping empty chunk {i} during processing")
            continue
        
        # Use i as the vector_number (actual position after filtering)
        vector_number = i
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")
        
        # Get heading from metadata, prefer section information over generic chunk names
        heading = chunk.metadata.get("heading", "")
        section_title = chunk.metadata.get("section_title", "")
        section_path = chunk.metadata.get("section_path", "")
        
        # If heading is empty or is a fallback pattern like "Chunk X", try to use section information
        if not heading or heading.startswith("Chunk "):
            # Prefer section_title, then last part of section_path, then generic name
            if section_title:
                heading = section_title
            elif section_path:
                path_parts = section_path.split(" > ")
                heading = path_parts[-1] if path_parts else f"Chunk {vector_number}"
            else:
                heading = f"Chunk {vector_number}"
        # If heading is a real heading (not "Chunk X" pattern), keep it as is
        
        section_path = chunk.metadata.get("section_path", "")
        
        # No chunk summary (not used downstream); keep key for compatibility
        summary = ""
        
        # Prepare content for embedding
        content = chunk.page_content
        original_length = len(content)
        content, was_truncated = token_tracker.check_embedding_limit(content)
        
        # Prepare metadata for Chroma
        prev_chunk_ids = chunk.metadata.get("prev_chunk_ids", [])
        next_chunk_ids = chunk.metadata.get("next_chunk_ids", [])
        page_number = chunk.metadata.get("page_number")
        page_classification = page_classifications.get(page_number) if page_number and page_number in page_classifications else ""
        
        chroma_metadata = {
            "vector_number": vector_number,
            "chunk_index": vector_number,
            "heading": heading or "",
            "summary": summary or "",
            "section_path": section_path or "",
            "section_title": chunk.metadata.get("section_title") or "",
            "section_level": int(chunk.metadata.get("section_level", 0)),
            "chunk_type": chunk.metadata.get("chunk_type", "text"),
            "table_context": chunk.metadata.get("table_context") or "",
            "original_length": int(original_length),
            "truncated": bool(was_truncated),
            "has_table": bool(chunk.metadata.get("has_table", False)),
            "page_number": int(page_number) if page_number is not None else 0,
            "page_classification": page_classification or "",
            "sequential_position": int(chunk.metadata.get("sequential_position", i)),
            "prev_chunk_ids": ",".join(map(str, prev_chunk_ids)) if prev_chunk_ids else "",
            "next_chunk_ids": ",".join(map(str, next_chunk_ids)) if next_chunk_ids else "",
            "start_line": int(chunk.metadata.get("start_line", 0)),
        }
        
        # Final verification
        if len(content) > EMBEDDING_MAX_CHARS:
            content = content[:EMBEDDING_MAX_CHARS] + "..."
            was_truncated = True
        
        final_tokens = token_tracker.count_tokens(content)
        if final_tokens > EMBEDDING_MAX_TOKENS:
            content = content[:int(EMBEDDING_MAX_CHARS * 0.8)] + "..."
            was_truncated = True
        
        # Create document for vector store
        doc = Document(
            page_content=content,
            metadata=chroma_metadata
        )
        
        # Add to vector store
        try:
            vector_store.add_documents([doc])
        except Exception as e:
            logger.error("Failed to add chunk %s: %s", vector_number, e)
            content = content[:int(EMBEDDING_MAX_CHARS * 0.5)] + "..."
            doc.page_content = content
            try:
                vector_store.add_documents([doc])
            except Exception as e2:
                logger.error("Still failed, skipping chunk %s: %s", vector_number, e2)
                continue
        
        # Add chunk node to graph
        chunk_node = document_graph.add_chunk_node(vector_number, doc)
        
        # Add edges: chunk belongs_to section
        if section_path:
            section_node = document_graph.section_nodes.get(section_path)
            if section_node:
                document_graph.add_edge(section_node, chunk_node, relation="contains")
                document_graph.add_edge(chunk_node, section_node, relation="belongs_to")
        
        # Add edges: chunk on_page page
        if page_number and page_number > 0:
            # Get classification for this page
            classification = page_classifications.get(page_number)
            page_node = document_graph.add_page_node(page_number, classification)
            if page_node:
                document_graph.add_edge(page_node, chunk_node, relation="on_page")
        
        # Add edges: chunk follows previous/next chunks
        for prev_id in prev_chunk_ids:
            prev_node = document_graph.chunk_nodes.get(prev_id)
            if prev_node:
                document_graph.add_edge(prev_node, chunk_node, relation="follows")
        
        for next_id in next_chunk_ids:
            next_node = document_graph.chunk_nodes.get(next_id)
            if next_node:
                document_graph.add_edge(chunk_node, next_node, relation="follows")
        
        # Create full metadata for JSON
        full_metadata = chroma_metadata.copy()
        full_metadata["prev_chunk_ids"] = prev_chunk_ids
        full_metadata["next_chunk_ids"] = next_chunk_ids
        
        chunk_detail = {
            "vector_number": vector_number,
            "heading": heading,
            "summary": summary,
            "section_path": section_path,
            "content": content,
            "content_length": len(content),
            "metadata": full_metadata
        }
        
        json_mapping.append(chunk_detail)
        processed_chunks.append(chunk_detail)
        token_tracker.stats["total_chunks"] += 1
    
    # Compute similarity edges between chunks (rank-based: top-K per chunk, embedding-model agnostic)
    logger.info("Computing similarity edges between chunks...")
    max_similar_per_chunk = 5  # Connect each chunk to its K nearest neighbors (by distance)
    min_similarity_for_edge = 0.5  # Only add edge if relative similarity > this (avoid weak/0.00 links)
    
    similarity_edges_added = 0
    all_chunk_ids = sorted(document_graph.chunk_nodes.keys())
    
    # Debug: track distance ranges across chunks (for logging only)
    distance_samples = []
    
    for i, chunk_id in enumerate(all_chunk_ids):
        if (i + 1) % 50 == 0:
            logger.info(f"  Computing similarities for chunk {i + 1}/{len(all_chunk_ids)}...")
        
        chunk_node = document_graph.chunk_nodes.get(chunk_id)
        if not chunk_node:
            continue
        
        # Get the chunk content from vector store
        try:
            # Use the chunk's content to find similar chunks
            chunk_doc = None
            for chunk_detail in processed_chunks:
                if chunk_detail.get("vector_number") == chunk_id:
                    chunk_doc = Document(
                        page_content=chunk_detail.get("content", ""),
                        metadata=chunk_detail.get("metadata", {})
                    )
                    break
            
            if not chunk_doc or not chunk_doc.page_content.strip():
                continue
            
            # Find similar chunks using vector similarity search
            # Search for more than we need to account for the chunk itself
            similar_results = vector_store.similarity_search_with_score(
                chunk_doc.page_content,
                k=max_similar_per_chunk + 3
            )
            
            if not similar_results:
                if i < 3:  # Debug for first few
                    logger.warning(f"  No results for chunk {chunk_id}")
                continue
            
            if i < 3:  # Debug for first few
                logger.info(f"  Chunk {chunk_id}: Found {len(similar_results)} search results")
            
            # Collect valid candidates: (chunk_id, distance, node); skip self, None, not in graph
            skipped_same = 0
            skipped_none = 0
            skipped_not_in_graph = 0
            candidates = []
            for similar_doc, distance_score in similar_results:
                similar_chunk_id = similar_doc.metadata.get("chunk_index")
                if similar_chunk_id is None:
                    skipped_none += 1
                    continue
                if similar_chunk_id == chunk_id:
                    skipped_same += 1
                    continue
                if similar_chunk_id not in document_graph.chunk_nodes:
                    skipped_not_in_graph += 1
                    continue
                similar_node = document_graph.chunk_nodes.get(similar_chunk_id)
                if similar_node:
                    candidates.append((similar_chunk_id, distance_score, similar_node))
            
            # Rank-based: take top max_similar_per_chunk by distance (lowest = most similar)
            candidates.sort(key=lambda x: x[1])
            top = candidates[:max_similar_per_chunk]
            
            if i < 3:
                logger.info(f"  Chunk {chunk_id}: {len(similar_results)} results, skipped: same={skipped_same}, none={skipped_none}, not_in_graph={skipped_not_in_graph}, top-K={len(top)}")
            
            # Relative similarity for edge weight: 1.0 = nearest, 0.0 = farthest in this top-K (embedding-model agnostic)
            d_min = top[0][1] if top else 0.0
            d_max = top[-1][1] if top else 0.0
            span = (d_max - d_min) + 1e-9
            
            for rank, (similar_chunk_id, distance_score, similar_node) in enumerate(top):
                # Relative similarity in [0, 1]: best=1.0, worst in top-K=0.0
                similarity = 1.0 - (distance_score - d_min) / span
                similarity = max(0.0, min(1.0, similarity))
                
                if len(distance_samples) < 20:
                    distance_samples.append((distance_score, similarity))
                if i < 3 and rank < 2:
                    logger.info(f"  Chunk {chunk_id} -> {similar_chunk_id}: distance={distance_score:.4f}, rel_similarity={similarity:.4f} (rank {rank+1}/{len(top)})")
                
                # Only add edge if relative similarity is above minimum (skip 0.00 or weak links)
                if similarity <= min_similarity_for_edge:
                    continue
                # Skip if already connected via "follows"
                existing_edge_1 = document_graph.graph.get_edge_data(chunk_node, similar_node)
                existing_edge_2 = document_graph.graph.get_edge_data(similar_node, chunk_node)
                has_follows = (existing_edge_1 and existing_edge_1.get("relation") == "follows") or \
                              (existing_edge_2 and existing_edge_2.get("relation") == "follows")
                if has_follows:
                    continue
                if (existing_edge_1 and existing_edge_1.get("relation") == "similar_to") or \
                   (existing_edge_2 and existing_edge_2.get("relation") == "similar_to"):
                    continue
                
                document_graph.add_edge(
                    chunk_node,
                    similar_node,
                    relation="similar_to",
                    similarity=similarity
                )
                similarity_edges_added += 1
                
        except Exception as e:
            logger.warning(f"Error computing similarity for chunk {chunk_id}: {e}", exc_info=True)
            continue
    
    # Log distance statistics for debugging (sample across chunks; scale is embedding-model dependent)
    if distance_samples:
        distances = [d[0] for d in distance_samples]
        rel_sims = [d[1] for d in distance_samples]
        logger.info(f"Distance sample (model-dependent): min={min(distances):.4f}, max={max(distances):.4f}, avg={sum(distances)/len(distances):.4f}")
        logger.info(f"Relative similarity (rank-based): min={min(rel_sims):.4f}, max={max(rel_sims):.4f}, avg={sum(rel_sims)/len(rel_sims):.4f}")
    
    logger.info(f"Added {similarity_edges_added} similarity edges between chunks")
    
    # Save files
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_mapping, f, indent=2, ensure_ascii=False)
    
    document_graph.save(graph_output_path)
    
    # Save page classifications if available
    if page_classifications:
        classifications_path = plan_e_dir / f"{doc_stem}_page_classifications.json"
        with open(classifications_path, 'w', encoding='utf-8') as f:
            json.dump(page_classifications, f, indent=2, ensure_ascii=False)
        logger.info(f"Page classifications saved to: {classifications_path}")
    
    state['processed_chunks'] = processed_chunks
    state['vector_store'] = vector_store
    state['document_graph'] = document_graph
    state['json_mapping'] = json_mapping
    
    # Log stats and attach token usage for economics tracking (main.py will persist to economics/)
    final_stats = token_tracker.get_stats()
    state['token_usage'] = final_stats
    logger.info("=" * 60)
    logger.info("Token Usage Statistics:")
    logger.info(f"  Total chunks processed: {final_stats['total_chunks']}")
    logger.info(f"  Total embedding tokens: {final_stats['embedding_tokens']}")
    logger.info(f"  Total LLM tokens: {final_stats['llm_tokens']}")
    logger.info(f"  Truncated chunks: {final_stats['truncated_chunks']}")
    logger.info("=" * 60)
    
    logger.info(f"Completed processing all {len(processed_chunks)} chunks")
    logger.info(f"Vector store saved to: {output_dir}")
    logger.info(f"JSON mapping saved to: {json_output_path}")
    logger.info(f"Document graph saved to: {graph_output_path}")
    logger.info(f"Graph: {len(document_graph.graph.nodes)} nodes, {len(document_graph.graph.edges)} edges")
    
    return state

# ---------------- LANGGRAPH WORKFLOW ---------------- 
def create_vectorization_workflow() -> StateGraph:
    """Create LangGraph workflow for vectorization"""
    workflow = StateGraph(VectorizerState)
    
    workflow.add_node("load_markdown", load_markdown)
    workflow.add_node("process_chunks", process_chunks_one_by_one)
    
    workflow.set_entry_point("load_markdown")
    workflow.add_edge("load_markdown", "process_chunks")
    workflow.add_edge("process_chunks", END)
    
    return workflow.compile()

# ---------------- MAIN FUNCTION ---------------- 
def main():
    """Main entry point"""
    logger.info("Checking inference backend...")
    is_running, available_models = check_inference_ready()
    if not is_running:
        logger.error("=" * 60)
        logger.error("Inference backend is not ready!")
        logger.error("=" * 60)
        logger.error("For Ollama: start with 'ollama serve' and pull models.")
        logger.error("For Hugging Face: set HUGGINGFACEHUB_API_TOKEN and INFERENCE_PROVIDER=huggingface")
        logger.error("=" * 60)
        sys.exit(1)
    logger.info("✓ Inference backend is ready: %s", available_models[:3] if len(available_models) > 3 else available_models)
    
    if TIKTOKEN_AVAILABLE:
        logger.info("✓ Token tracking enabled with tiktoken")
    else:
        logger.warning("⚠ tiktoken not available - using fallback token counting")
    
    if not NETWORKX_AVAILABLE:
        logger.error("=" * 60)
        logger.error("networkx is required for Plan E!")
        logger.error("=" * 60)
        logger.error("Install with: pip install networkx")
        logger.error("=" * 60)
        sys.exit(1)
    logger.info("✓ NetworkX available for graph structure")
    
    # Accept folder name or file path as argument
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # Default: look for folders in output directory
        output_dir = Path("output")
        if output_dir.exists():
            # Get most recent folder
            folders = [d for d in output_dir.iterdir() if d.is_dir()]
            if folders:
                input_path = max(folders, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent folder: {input_path}")
            else:
                logger.error("No folders found in output directory. Please specify a folder or file path.")
                sys.exit(1)
        else:
            logger.error("Output directory not found. Please specify a folder or file path.")
            sys.exit(1)
    
    # If it's a folder, use it directly; if it's a file, use its parent folder
    if input_path.is_file():
        root_folder = input_path.parent
        logger.info(f"File provided, using folder: {root_folder}")
    elif input_path.is_dir():
        root_folder = input_path
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("VectorizerE - Plan E: Graph-Based Document Structure")
    logger.info("=" * 60)
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Embedding model: {get_embedding_model_id()}")
    logger.info(f"LLM model: {get_llm_model_id()}")
    logger.info(f"Token tracking: {'Enabled' if TIKTOKEN_AVAILABLE else 'Fallback'}")
    logger.info("=" * 60)

    # Determine if this is a single-document folder or a collection of document folders
    md_files_in_root = list(root_folder.glob("*.md"))
    subfolders = [d for d in root_folder.iterdir() if d.is_dir()]

    # Helper to run workflow on a single document folder
    def run_for_doc_folder(doc_folder: Path):
        md_files = list(doc_folder.glob("*.md"))
        if not md_files:
            logger.warning(f"No .md file found in folder: {doc_folder}, skipping")
            return
        input_file = md_files[0]

        logger.info("=" * 60)
        logger.info(f"Processing document folder: {doc_folder}")
        logger.info(f"Markdown file: {input_file.name}")
        logger.info(f"Plan E output folder: {doc_folder / 'E'}")
        logger.info("=" * 60)

        initial_state: VectorizerState = {
            "markdown_file": str(doc_folder),  # Pass folder path
            "chunks": [],
            "structure": {},
            "processed_chunks": [],
            "vector_store": None,
            "document_graph": DocumentGraph(),
            "json_mapping": [],
            "page_mapping": None,
            "page_classifications": None,
            "output_folder": str(doc_folder)
        }

        workflow = create_vectorization_workflow()
        final_state = workflow.invoke(initial_state)

        logger.info("=" * 60)
        logger.info("Vectorization Complete for this document!")
        logger.info("=" * 60)
        logger.info(f"Total vectors created: {len(final_state['json_mapping'])}")
        output_folder = Path(final_state.get("output_folder", doc_folder))
        logger.info(f"Plan E output folder: {output_folder / 'E'}")
        doc_stem = input_file.stem
        logger.info(f"Vector DB location: {output_folder / 'E' / 'vector_db' / doc_stem}")
        logger.info(f"JSON mapping: {output_folder / 'E' / f'{doc_stem}_vector_mapping.json'}")
        logger.info(f"Document graph: {output_folder / 'E' / f'{doc_stem}_document_graph.json'}")
        logger.info("=" * 60)
        logger.info("")

    try:
        # Mode 1: root folder itself contains a single markdown (single document)
        if md_files_in_root:
            run_for_doc_folder(root_folder)
        # Mode 2: root folder contains multiple subfolders, each a document folder
        elif subfolders:
            logger.info(f"Detected {len(subfolders)} document subfolders under {root_folder}")
            for doc_folder in sorted(subfolders):
                run_for_doc_folder(doc_folder)
            logger.info("All document folders processed.")
        else:
            logger.error(f"No markdown files or subfolders found in {root_folder}")
            sys.exit(1)

        logger.info("Graph-based retrieval is now available for processed documents!")
        logger.info("Use DocumentGraph.expand_from_chunks() for graph-enhanced retrieval")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during vectorization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
