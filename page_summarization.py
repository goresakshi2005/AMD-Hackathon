"""
Page Summarization Agent
Generates comprehensive page-level explanations using LLM.
Handles cases where pages have no chunks by analyzing adjacent pages.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

from config.inference_config import get_llm
from retrivalAgentE import (
    DocumentGraph,
    load_chunks_from_mapping,
    find_vector_mapping_file,
    find_graph_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


@dataclass
class PageSummary:
    """Page summary result"""
    page_number: int
    summary: str
    key_points: List[str]
    sections: List[str]
    has_content: bool
    used_adjacent_pages: bool
    adjacent_pages_used: Optional[List[int]] = None
    page_classification: Optional[str] = None
    chunks_used: List[int] = None


class PageSummarizationAgent:
    """Agent for generating page-level summaries"""
    
    def __init__(self, document_graph: DocumentGraph, chunks: List[Document], llm: Any):
        self.document_graph = document_graph
        self.chunks = chunks
        self.llm = llm
        self.chunk_dict = {chunk.metadata.get("chunk_index"): chunk for chunk in chunks}
    
    def get_chunks_by_page(self, page_number: int) -> List[Document]:
        """Get all chunks on a specific page"""
        chunk_ids = self.document_graph.get_page_chunks(page_number)
        chunks = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_dict:
                chunks.append(self.chunk_dict[chunk_id])
        return chunks
    
    def get_chunks_by_page_range(self, start_page: int, end_page: int) -> List[Document]:
        """Get all chunks from a range of pages"""
        all_chunks = []
        for page_num in range(start_page, end_page + 1):
            chunks = self.get_chunks_by_page(page_num)
            all_chunks.extend(chunks)
        return all_chunks
    
    def get_adjacent_page_chunks(self, page_number: int, window: int = 1) -> Tuple[List[Document], List[int]]:
        """Get chunks from adjacent pages (previous and next)"""
        prev_page = max(1, page_number - window)
        next_page = page_number + window
        
        prev_chunks = self.get_chunks_by_page_range(prev_page, page_number - 1) if prev_page < page_number else []
        next_chunks = self.get_chunks_by_page_range(page_number + 1, next_page) if next_page > page_number else []
        
        all_adjacent = prev_chunks + next_chunks
        pages_used = []
        
        if prev_chunks:
            pages_used.extend(range(prev_page, page_number))
        if next_chunks:
            pages_used.extend(range(page_number + 1, next_page + 1))
        
        return all_adjacent, pages_used
    
    def get_page_classification(self, page_number: int) -> Optional[str]:
        """Get page classification if available"""
        page_node = self.document_graph.page_nodes.get(page_number)
        if page_node:
            return self.document_graph.graph.nodes[page_node].get("classification")
        return None
    
    def summarize_page(self, page_number: int, use_adjacent_if_empty: bool = True) -> PageSummary:
        """
        Generate comprehensive page summary
        
        Args:
            page_number: Page number to summarize
            use_adjacent_if_empty: If True, use adjacent pages when current page has no chunks
        
        Returns:
            PageSummary object with page explanation
        """
        logger.info(f"Summarizing page {page_number}...")
        
        # Get chunks for this page
        page_chunks = self.get_chunks_by_page(page_number)
        used_adjacent = False
        adjacent_pages = None
        
        # If no chunks and use_adjacent_if_empty, get adjacent pages
        if not page_chunks and use_adjacent_if_empty:
            logger.info(f"Page {page_number} has no chunks, using adjacent pages...")
            adjacent_chunks, adjacent_pages = self.get_adjacent_page_chunks(page_number, window=1)
            
            if adjacent_chunks:
                page_chunks = adjacent_chunks
                used_adjacent = True
                logger.info(f"Using {len(adjacent_chunks)} chunks from adjacent pages: {adjacent_pages}")
            else:
                # Try wider window
                adjacent_chunks, adjacent_pages = self.get_adjacent_page_chunks(page_number, window=2)
                if adjacent_chunks:
                    page_chunks = adjacent_chunks
                    used_adjacent = True
                    logger.info(f"Using {len(adjacent_chunks)} chunks from wider adjacent pages: {adjacent_pages}")
        
        # Get page classification
        page_classification = self.get_page_classification(page_number)
        
        # Get chunk IDs used
        chunk_ids = [chunk.metadata.get("chunk_index") for chunk in page_chunks if chunk.metadata.get("chunk_index") is not None]
        
        if not page_chunks:
            logger.warning(f"No chunks found for page {page_number} even with adjacent pages")
            return PageSummary(
                page_number=page_number,
                summary="No content available for this page. The page may be blank, contain only images, or have insufficient text for processing.",
                key_points=[],
                sections=[],
                has_content=False,
                used_adjacent_pages=False,
                page_classification=page_classification,
                chunks_used=[],
            )
        
        # Combine chunk content
        page_content = "\n\n".join([chunk.page_content for chunk in page_chunks])
        
        # Extract sections from chunks
        sections = []
        seen_sections = set()
        for chunk in page_chunks:
            section_path = chunk.metadata.get("section_path", "")
            if section_path and section_path not in seen_sections:
                sections.append(section_path)
                seen_sections.add(section_path)
        
        # Generate summary using LLM
        summary, key_points = self._generate_summary_with_llm(
            page_number,
            page_content,
            page_classification,
            used_adjacent,
            adjacent_pages
        )
        
        return PageSummary(
            page_number=page_number,
            summary=summary,
            key_points=key_points,
            sections=sections,
            has_content=True,
            used_adjacent_pages=used_adjacent,
            adjacent_pages_used=adjacent_pages if used_adjacent else None,
            page_classification=page_classification,
            chunks_used=chunk_ids,
        )
    
    def _generate_summary_with_llm(
        self,
        page_number: int,
        content: str,
        classification: Optional[str],
        used_adjacent: bool,
        adjacent_pages: Optional[List[int]]
    ) -> Tuple[str, List[str]]:
        """Generate page summary and key points using LLM"""
        
        # Truncate content if too long (keep first 4000 chars for LLM)
        content_for_llm = content[:4000] + "..." if len(content) > 4000 else content
        
        context_note = ""
        if used_adjacent and adjacent_pages:
            context_note = f"\n\nNOTE: Page {page_number} had no direct content, so this summary is based on content from adjacent pages: {', '.join(map(str, adjacent_pages))}."
        
        classification_note = f"\n\nPage Classification: {classification}" if classification else ""
        
        prompt = f"""You are analyzing page {page_number} of a document.

Page Content:
{content_for_llm}
{classification_note}
{context_note}

Your task:
1. Provide a comprehensive summary of this page (3-5 sentences)
2. Extract 3-7 key points or important information from this page
3. If the page appears to be a continuation or part of a larger section, note that in the summary
4. Be specific and detailed - this summary will be used to explain the page to users

Format your response EXACTLY as follows:
SUMMARY: [Your comprehensive summary here]

KEY_POINTS:
- [First key point]
- [Second key point]
- [Third key point]
[Continue with more key points as needed]

Now provide the summary and key points:"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = (getattr(response, "content", None) or str(response)).strip()
            
            # Parse response
            summary = ""
            key_points = []
            
            if "SUMMARY:" in response_text:
                summary_part = response_text.split("SUMMARY:")[1]
                if "KEY_POINTS:" in summary_part:
                    summary = summary_part.split("KEY_POINTS:")[0].strip()
                    key_points_part = summary_part.split("KEY_POINTS:")[1]
                else:
                    summary = summary_part.strip()
            else:
                # Fallback: use first paragraph as summary
                lines = response_text.split("\n")
                summary = lines[0] if lines else response_text[:200]
            
            # Extract key points
            if "KEY_POINTS:" in response_text:
                key_points_section = response_text.split("KEY_POINTS:")[1]
                for line in key_points_section.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("*"):
                        point = line.lstrip("-* ").strip()
                        if point:
                            key_points.append(point)
                    elif line and not line.startswith("SUMMARY:"):
                        key_points.append(line)
            
            # If no key points extracted, create from summary
            if not key_points:
                # Split summary into sentences and use first few as key points
                sentences = summary.split(". ")
                key_points = [s.strip() + "." for s in sentences[:5] if s.strip()]
            
            # Clean up summary
            summary = summary.strip()
            if not summary:
                summary = "This page contains content that requires analysis. Please refer to the document for details."
            
            return summary, key_points[:7]  # Limit to 7 key points
            
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {e}", exc_info=True)
            # Fallback summary
            fallback_summary = f"This page contains document content. "
            if classification:
                fallback_summary += f"It is classified as: {classification}. "
            if used_adjacent:
                fallback_summary += "Summary based on adjacent pages."
            else:
                fallback_summary += "Please refer to the document for full details."
            
            # Extract key points from content (first few sentences)
            sentences = content.split(". ")[:5]
            key_points = [s.strip() + "." for s in sentences if len(s.strip()) > 20]
            
            return fallback_summary, key_points


def load_page_agent(document_folder: Path, llm: Any = None) -> Optional[PageSummarizationAgent]:
    """Load page summarization agent for a document. Uses provided llm or get_llm() from inference_config."""
    plan_e_dir = document_folder / "E"
    
    if not plan_e_dir.exists():
        logger.error(f"Plan E folder not found: {plan_e_dir}")
        return None
    
    # Find required files
    vector_mapping_file = find_vector_mapping_file(plan_e_dir)
    graph_file = find_graph_file(plan_e_dir)
    
    if not vector_mapping_file or not vector_mapping_file.exists():
        logger.error("Vector mapping file not found")
        return None
    
    if not graph_file or not graph_file.exists():
        logger.error("Graph file not found")
        return None
    
    # Load chunks and graph
    chunks = load_chunks_from_mapping(vector_mapping_file)
    if not chunks:
        logger.error("Failed to load chunks")
        return None
    
    document_graph = DocumentGraph()
    document_graph.load(graph_file)
    
    if llm is None:
        llm = get_llm(temperature=0.3)
    
    return PageSummarizationAgent(document_graph, chunks, llm)


def summarize_page_for_document(document_folder: Path, page_number: int) -> Dict[str, Any]:
    """Convenience function to summarize a page"""
    agent = load_page_agent(document_folder)
    if not agent:
        raise ValueError("Failed to load page summarization agent")
    
    summary = agent.summarize_page(page_number)
    
    # Convert to dict
    return {
        "page_number": summary.page_number,
        "summary": summary.summary,
        "key_points": summary.key_points,
        "sections": summary.sections,
        "has_content": summary.has_content,
        "used_adjacent_pages": summary.used_adjacent_pages,
        "adjacent_pages_used": summary.adjacent_pages_used,
        "page_classification": summary.page_classification,
        "chunks_used": summary.chunks_used,
        "total_chunks": len(summary.chunks_used),
    }


if __name__ == "__main__":
    """Test the page summarization"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python page_summarization.py <document_folder> <page_number>")
        sys.exit(1)
    
    doc_folder = Path(sys.argv[1])
    page_num = int(sys.argv[2])
    
    result = summarize_page_for_document(doc_folder, page_num)
    
    print("\n" + "=" * 80)
    print(f"PAGE {page_num} SUMMARY")
    print("=" * 80)
    print(f"\nClassification: {result.get('page_classification', 'N/A')}")
    print(f"Has Content: {result.get('has_content', False)}")
    print(f"Used Adjacent Pages: {result.get('used_adjacent_pages', False)}")
    if result.get('adjacent_pages_used'):
        print(f"Adjacent Pages Used: {result['adjacent_pages_used']}")
    print(f"Total Chunks: {result.get('total_chunks', 0)}")
    
    print(f"\nSections:")
    for section in result.get('sections', []):
        print(f"  - {section}")
    
    print(f"\nSummary:")
    print(result.get('summary', 'N/A'))
    
    print(f"\nKey Points:")
    for point in result.get('key_points', []):
        print(f"  • {point}")
    
    print("=" * 80)
