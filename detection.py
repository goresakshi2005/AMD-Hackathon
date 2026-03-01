from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from hierarchical.postprocessor import ResultPostprocessor
import logging
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress RapidOCR warnings about empty results (these are harmless when PDF has native text)
logging.getLogger("RapidOCR").setLevel(logging.ERROR)

# ---------------- MARKDOWN FIXING HELPERS ----------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def is_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"[\d.,%-]+", normalize(text)))

def is_header_like(row):
    """
    Header rows are mostly non-numeric.
    """
    non_numeric = sum(not is_numeric(c) for c in row if normalize(c))
    return non_numeric / max(len(row), 1) > 0.7

def is_group_header(row):
    """
    Example:
    | 5. | Course Contents (Semester V) |   |
    """
    if len(row) < 3:
        return False
    return (
        re.fullmatch(r"\d+\.", row[0].strip()) and
        normalize(row[-1]) == "" and
        normalize(row[1]) != ""
    )

def is_child_row(row):
    return normalize(row[0]) == "" and normalize(row[-1]) != ""

def has_only_one_filled_cell(row):
    """
    Check if a row has only one non-empty cell.
    Returns True if exactly one cell has content, False otherwise.
    For header-like rows, check after applying blank_repeated_adjacent_columns.
    """
    # If it's header-like, check what it would become after transformation
    if is_header_like(row):
        transformed = blank_repeated_adjacent_columns(row)
        filled_count = sum(1 for c in transformed if normalize(c) != "")
        return filled_count == 1
    else:
        filled_count = sum(1 for c in row if normalize(c) != "")
        return filled_count == 1

def blank_repeated_adjacent_columns(row):
    """
    For header-like rows:
    keep first occurrence, blank consecutive duplicates.
    """
    new_row = []
    prev = None

    for cell in row:
        if prev is not None and normalize(cell) == normalize(prev):
            new_row.append("")
        else:
            new_row.append(cell)
            prev = cell

    return new_row

def is_separator_row(row):
    """
    Check if a row is a markdown table separator (all dashes or mostly dashes).
    """
    if not row:
        return False
    # Check if all cells are mostly dashes or empty
    dash_count = 0
    total_chars = 0
    for cell in row:
        cell_text = normalize(cell)
        if cell_text:
            total_chars += len(cell_text)
            dash_count += cell_text.count("-")
    
    if total_chars == 0:
        return False
    
    # If more than 80% of characters are dashes, it's a separator row
    return dash_count / total_chars > 0.8

def trim_trailing_empty_columns(rows):
    """
    Remove trailing empty columns from all rows.
    """
    if not rows:
        return rows
    
    # Find the rightmost column that has at least one non-empty cell
    max_cols = max(len(row) for row in rows) if rows else 0
    rightmost_filled = -1
    
    for col_idx in range(max_cols - 1, -1, -1):
        for row in rows:
            if col_idx < len(row) and normalize(row[col_idx]) != "":
                rightmost_filled = col_idx
                break
        if rightmost_filled >= 0:
            break
    
    # Trim all rows to the rightmost filled column + 1
    if rightmost_filled >= 0:
        return [row[:rightmost_filled + 1] for row in rows]
    return rows

# ---------------- TABLE FIX ----------------
def fix_table(table_lines):
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    # Split tables at rows with only one filled cell
    table_groups = []
    current_group = []
    
    for i, row in enumerate(rows):
        # Skip separator rows when splitting (they'll be handled later)
        if is_separator_row(row):
            current_group.append(row)
            continue
            
        # Check if this row has only one unique filled element
        # (either one filled cell, or all filled cells have the same normalized content)
        filled_cells = [normalize(c) for c in row if normalize(c) != ""]
        unique_filled = len(set(filled_cells))
        
        # If this row has only one unique filled element and it's not the first row,
        # split the table here - treat it as a standalone table
        if i > 0 and unique_filled == 1:
            if current_group:
                table_groups.append(current_group)
            current_group = [row]
        else:
            current_group.append(row)
    
    # Add the last group
    if current_group:
        table_groups.append(current_group)
    
    # Process each table group
    all_tables = []
    for group in table_groups:
        fixed = []
        last_was_group = False
        has_separator = False

        # Process rows: convert separator rows and apply transformations
        for row in group:
            if is_separator_row(row):
                has_separator = True
                # Don't add separator row here, we'll add it after header
                continue
                
            # Group header
            if is_group_header(row):
                fixed.append(row)
                last_was_group = True
                continue

            # Child rows under group header
            if last_was_group and is_child_row(row):
                row[1] = "↳ " + row[1]
                fixed.append(row)
                continue

            last_was_group = False

            # Header-only cleanup
            if is_header_like(row):
                row = blank_repeated_adjacent_columns(row)

            fixed.append(row)

        # Trim trailing empty columns from all rows
        fixed = trim_trailing_empty_columns(fixed)
        
        if not fixed:
            continue
        
        # Determine number of columns (use max across all rows)
        num_cols = max(len(row) for row in fixed) if fixed else 0
        
        if num_cols == 0:
            continue
        
        # Ensure all rows have the same number of columns
        for row in fixed:
            while len(row) < num_cols:
                row.append("")
        
        # Rebuild markdown for this table
        out = []
        separator_added = False
        
        for idx, row in enumerate(fixed):
            line = "| " + " | ".join(row) + " |"
            out.append(line)
            
            # Add separator row after first row
            # Always add separator for proper markdown table rendering
            if idx == 0 and not separator_added:
                separator_line = "| " + " | ".join(["---"] * num_cols) + " |"
                out.append(separator_line)
                separator_added = True
        
        all_tables.append(out)
    
    return all_tables

# ---------------- FILE PROCESSOR ----------------
def process_markdown(text, preserve_page_breaks: bool = True):
    """
    Process markdown and fix tables while preserving page break markers.
    
    Args:
        text: Markdown text to process
        preserve_page_breaks: If True, preserve page break markers during processing
    """
    PAGE_BREAK_MARKER = "<!-- page break -->"
    lines = text.splitlines()
    output = []
    buffer = []
    in_table = False

    for line in lines:
        # Check for page break marker
        has_page_break = PAGE_BREAK_MARKER in line if preserve_page_breaks else False
        
        if line.strip().startswith("|") and "|" in line:
            buffer.append(line)
            in_table = True
        else:
            if in_table:
                tables = fix_table(buffer)
                # Add each table, separated by blank lines
                for i, table in enumerate(tables):
                    output.extend(table)
                    # Add blank line between tables (but not after the last one)
                    if i < len(tables) - 1:
                        output.append("")
                buffer = []
                in_table = False
            
            # Preserve page break markers
            if has_page_break:
                output.append(PAGE_BREAK_MARKER)
            else:
                output.append(line)

    if buffer:
        tables = fix_table(buffer)
        # Add each table, separated by blank lines
        for i, table in enumerate(tables):
            output.extend(table)
            # Add blank line between tables (but not after the last one)
            if i < len(tables) - 1:
                output.append("")

    return "\n".join(output)

# ---------------- PAGE MAPPING EXTRACTION ----------------
def extract_page_mapping_from_markdown(markdown_with_breaks: str) -> Dict[str, Any]:
    """
    Extract page boundaries from markdown that contains page break placeholders.
    Uses docling's page_break_placeholder feature to accurately map lines to pages.
    
    Returns:
        Dictionary with:
        - line_to_page: List mapping line index (0-based) to page number
        - page_boundaries: List of (start_line, end_line, page_number) tuples
        - total_pages: Total number of pages
    """
    logger.info("Extracting page boundaries from markdown with page break markers...")
    
    # Page break placeholder used by docling
    PAGE_BREAK_MARKER = "<!-- page break -->"
    
    lines = markdown_with_breaks.splitlines()
    line_to_page = []
    page_boundaries = []
    current_page = 1
    page_start_line = 0
    
    for line_idx, line in enumerate(lines):
        # Check if this line contains a page break marker
        if PAGE_BREAK_MARKER in line:
            # This line marks the end of current page and start of next page
            # The line itself belongs to the current page
            line_to_page.append(current_page)
            
            # Save boundary for current page (if it has content)
            if page_start_line < line_idx:
                page_boundaries.append((page_start_line, line_idx, current_page))
            
            # Move to next page
            current_page += 1
            page_start_line = line_idx + 1
        else:
            # Regular line - belongs to current page
            line_to_page.append(current_page)
    
    # Add final page boundary
    if page_start_line < len(lines):
        page_boundaries.append((page_start_line, len(lines) - 1, current_page))
    
    total_pages = current_page
    
    logger.info(f"Extracted page mapping: {total_pages} pages, {len(page_boundaries)} boundaries")
    
    return {
        "line_to_page": line_to_page,
        "page_boundaries": page_boundaries,
        "total_pages": total_pages
    }

def extract_page_mapping_page_by_page(doc) -> Dict[str, Any]:
    """
    Extract page mapping by processing document page by page.
    This is the most accurate method as it uses docling's native page structure.
    
    Returns:
        Dictionary with:
        - page_contents: List of markdown content for each page
        - total_pages: Total number of pages
    """
    logger.info("Extracting page mapping using page-by-page approach...")
    
    try:
        # Get all pages from the document
        # Docling stores pages in doc.document structure
        pages_content = []
        total_pages = 0
        
        # Method 1: Try to access pages directly if available
        if hasattr(doc.document, 'pages') and doc.document.pages:
            logger.info("Found pages attribute in document")
            for page in doc.document.pages:
                total_pages += 1
                # Export this page to markdown
                page_md = page.export_to_markdown() if hasattr(page, 'export_to_markdown') else ""
                pages_content.append(page_md)
        
        # Method 2: Use text elements grouped by page
        elif hasattr(doc.document, 'texts'):
            logger.info("Grouping text elements by page...")
            pages_dict = {}  # page_no -> list of text items
            
            for text_item in doc.document.texts:
                if hasattr(text_item, 'prov') and text_item.prov:
                    for prov in text_item.prov:
                        if hasattr(prov, 'page_no'):
                            page_no = prov.page_no
                            if page_no not in pages_dict:
                                pages_dict[page_no] = []
                            pages_dict[page_no].append(text_item.text)
                            break
            
            # Sort by page number and create markdown for each page
            for page_no in sorted(pages_dict.keys()):
                page_texts = pages_dict[page_no]
                page_content = "\n".join(page_texts)
                pages_content.append(page_content)
                total_pages = max(total_pages, page_no)
        
        if pages_content:
            logger.info(f"Extracted {total_pages} pages using page-by-page method")
            return {
                "page_contents": pages_content,
                "total_pages": total_pages,
                "method": "page_by_page"
            }
        else:
            logger.warning("Could not extract pages using page-by-page method")
            return None
            
    except Exception as e:
        logger.warning(f"Error in page-by-page extraction: {e}")
        return None

def create_approximate_page_mapping(markdown_lines: List[str]) -> Dict[str, Any]:
    """
    Create approximate page mapping when page information is not available.
    Assumes ~50 lines per page (adjustable based on document type).
    """
    logger.info("Creating approximate page mapping (50 lines per page)")
    lines_per_page = 50
    
    line_to_page = []
    page_boundaries = []
    total_pages = (len(markdown_lines) // lines_per_page) + 1
    
    for line_idx in range(len(markdown_lines)):
        page_num = (line_idx // lines_per_page) + 1
        line_to_page.append(page_num)
        
        # Track page boundaries
        if line_idx == 0 or line_to_page[line_idx - 1] != page_num:
            if line_idx > 0:
                page_boundaries.append((line_idx - lines_per_page, line_idx - 1, page_num - 1))
            page_boundaries.append((line_idx, min(line_idx + lines_per_page - 1, len(markdown_lines) - 1), page_num))
    
    return {
        "line_to_page": line_to_page,
        "page_boundaries": page_boundaries,
        "total_pages": total_pages
    }

def process_single_pdf(pdf_path: Path, base_output_dir: Path, target_dir: Optional[Path] = None):
    """
    Process a single PDF:
    - Convert to markdown
    - Fix tables
    - Add page break markers
    - Generate page mapping JSON
    
    Args:
        pdf_path: Path to the PDF file
        base_output_dir: Base output directory (used if target_dir is None)
        target_dir: Optional target directory. If provided, files are saved here instead of base_output_dir/{pdf_stem}
    """
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return

    logger.info(f"Processing file: {pdf_path}")
    logger.info(f"File type: {pdf_path.suffix.upper() if pdf_path.suffix else 'Unknown'}")

    # Use target_dir if provided, otherwise create per-document output directory: {base_output_dir}/{pdf_stem}
    if target_dir is not None:
        doc_output_dir = Path(target_dir)
    else:
        input_stem = pdf_path.stem
        doc_output_dir = base_output_dir / input_stem
    
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = pdf_path.stem
    md_path = doc_output_dir / f"{input_stem}.md"
    page_mapping_path = doc_output_dir / f"{input_stem}_page_mapping.json"
    confidence_path = doc_output_dir / f"{input_stem}_confidence.json"

    logger.info("Initializing Docling converter")
    
    # Configure Docling to prefer native text extraction and avoid unnecessary OCR
    # This prevents OCR from being triggered for PDFs that already have selectable text
    # Setting do_ocr=False disables OCR entirely and uses only native text extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Disable OCR - use native text only
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    logger.info("✓ Configured to use native text extraction (OCR disabled)")

    logger.info(f"Converting document: {pdf_path.name}")
    doc = converter.convert(str(pdf_path))

    # ---------------- CONFIDENCE / ACCURACY SCORES ----------------
    # Docling exposes conversion quality via `ConversionResult.confidence`.
    # This contains numeric scores (0.0–1.0) and categorical grades such as
    # POOR / FAIR / GOOD / EXCELLENT, both at document-level and per-page.
    confidence = getattr(doc, "confidence", None)
    if confidence is not None:
        # Document-level scores
        layout_score = getattr(confidence, "layout_score", None)
        ocr_score = getattr(confidence, "ocr_score", None)
        parse_score = getattr(confidence, "parse_score", None)
        table_score = getattr(confidence, "table_score", None)

        mean_grade = getattr(confidence, "mean_grade", None)
        low_grade = getattr(confidence, "low_grade", None)

        logger.info("Docling conversion confidence (document-level):")
        logger.info(
            "  Scores - layout: %s, ocr: %s, parse: %s, table: %s",
            layout_score,
            ocr_score,
            parse_score,
            table_score,
        )
        logger.info("  Grades - mean: %s, low: %s", mean_grade, low_grade)

        # Page-level scores (one line per page)
        pages_conf = getattr(confidence, "pages", None)
        if isinstance(pages_conf, dict) and pages_conf:
            logger.info("Docling conversion confidence (per-page scores):")
            for page_no, page_scores in sorted(pages_conf.items()):
                p_layout = getattr(page_scores, "layout_score", None)
                p_ocr = getattr(page_scores, "ocr_score", None)
                p_parse = getattr(page_scores, "parse_score", None)
                p_table = getattr(page_scores, "table_score", None)
                logger.info(
                    "  Page %s - layout: %s, ocr: %s, parse: %s, table: %s",
                    page_no,
                    p_layout,
                    p_ocr,
                    p_parse,
                    p_table,
                )

        # Persist full confidence report so the API / UI can render it
        try:
            if hasattr(confidence, "model_dump_json"):
                confidence_data = json.loads(confidence.model_dump_json())
            elif hasattr(confidence, "dict"):
                confidence_data = confidence.dict()
            else:
                confidence_data = json.loads(json.dumps(confidence, default=lambda o: o.__dict__))

            with open(confidence_path, "w", encoding="utf-8") as f:
                json.dump(confidence_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Confidence report saved to: {confidence_path}")
        except Exception as e:
            logger.warning(f"Failed to save confidence report: {e}")

    # ---------------- APPLY HIERARCHICAL POSTPROCESSOR ----------------
    logger.info("Applying hierarchical postprocessor to fix heading levels...")
    try:
        # Apply the postprocessor to infer and correct heading hierarchy
        # Pass source path explicitly to ensure TOC/bookmark extraction works
        ResultPostprocessor(doc, source=str(pdf_path)).process()
        logger.info("✓ Successfully applied hierarchical postprocessor")
    except Exception as e:
        logger.warning(f"Hierarchical postprocessor failed: {e}")
        logger.info("Continuing without hierarchical postprocessing...")
        # Try without explicit source parameter as fallback
        try:
            ResultPostprocessor(doc).process()
            logger.info("✓ Applied hierarchical postprocessor (without explicit source)")
        except Exception as e2:
            logger.warning(f"Hierarchical postprocessor failed even without source: {e2}")
            logger.info("Proceeding with original heading levels...")

    # ---------------- EXPORT MARKDOWN WITH PAGE BREAKS ----------------
    logger.info("Exporting structured Markdown with page break markers")

    # Try method 1: Use page_break_placeholder (docling 2.28.2+)
    markdown = None
    page_mapping = None

    try:
        markdown = doc.document.export_to_markdown(page_break_placeholder="<!-- page break -->")
        logger.info("✓ Successfully exported markdown with page break markers (using page_break_placeholder)")
    except TypeError as e:
        logger.warning(f"page_break_placeholder not supported: {e}")
        logger.info("Trying page-by-page extraction method...")
        
        # Method 2: Extract page-by-page and reconstruct
        page_by_page_data = extract_page_mapping_page_by_page(doc)
        
        if page_by_page_data and page_by_page_data.get("page_contents"):
            # Reconstruct markdown with page breaks between pages
            page_contents = page_by_page_data["page_contents"]
            markdown = "\n<!-- page break -->\n".join(page_contents)
            logger.info(f"✓ Reconstructed markdown with {page_by_page_data['total_pages']} pages (page-by-page method)")
        else:
            # Method 3: Fallback - export normally and use text element grouping
            logger.warning("Page-by-page extraction failed, using text element grouping...")
            markdown = doc.document.export_to_markdown()
            
            # Group text elements by page and insert markers (heuristic)
            try:
                pages_dict = {}  # page_no -> list of text items
                for text_item in doc.document.texts:
                    if hasattr(text_item, 'prov') and text_item.prov:
                        for prov in text_item.prov:
                            if hasattr(prov, 'page_no'):
                                page_no = prov.page_no
                                if page_no not in pages_dict:
                                    pages_dict[page_no] = []
                                pages_dict[page_no].append(text_item.text)
                                break
                
                if pages_dict:
                    logger.info(f"Found {len(pages_dict)} pages via text element grouping")
            except Exception as e2:
                logger.warning(f"Text element grouping failed: {e2}")

    # If markdown is still None, use basic export
    if markdown is None:
        logger.warning("All methods failed, using basic markdown export")
        markdown = doc.document.export_to_markdown()

    # ---------------- FIX MARKDOWN TABLES ----------------
    logger.info("Fixing markdown table formatting (preserving page breaks)")
    if markdown:
        fixed_markdown = process_markdown(markdown, preserve_page_breaks=True)
    else:
        fixed_markdown = ""

    # ---------------- EXTRACT PAGE MAPPING ----------------
    # Extract page mapping from markdown with page break markers
    page_mapping = extract_page_mapping_from_markdown(fixed_markdown)

    # If page mapping failed (no markers found), try fallback
    if not page_mapping or page_mapping.get("total_pages", 0) == 0:
        logger.warning("No page break markers found, using approximate mapping")
        markdown_lines = fixed_markdown.splitlines()
        page_mapping = create_approximate_page_mapping(markdown_lines)

    # ---------------- SAVE FIXED MARKDOWN ----------------
    md_path.write_text(fixed_markdown, encoding="utf-8")

    # ---------------- SAVE PAGE MAPPING ----------------
    with open(page_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(page_mapping, f, indent=2, ensure_ascii=False)

    logger.info("DONE")
    logger.info(f"Output directory: {doc_output_dir}")
    logger.info(f"Fixed markdown saved to: {md_path}")
    logger.info(f"Page mapping saved to: {page_mapping_path}")
    logger.info(f"Total pages detected: {page_mapping['total_pages']}")


def main():
    """
    Entry point.

    Behaviours:
    - If given a PDF file path: process that single PDF into output/{pdf_stem}/
    - If given a folder path: process ALL *.pdf in that folder into
      output/{folder_stem}/{pdf_stem}/
    """
    # Accept file or folder from CLI, or default to single demo file
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path("invoice4.pdf")

    if not input_path.exists():
        print(f"Error: Path '{input_path}' not found!")
        sys.exit(1)

    # Base output directory
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    if input_path.is_file():
        # Single PDF mode: output/{pdf_stem}/...
        base_output_dir = out_dir
        process_single_pdf(input_path, base_output_dir)
    elif input_path.is_dir():
        # Folder mode: treat as a collection of PDFs
        pdf_files = sorted(list(input_path.glob("*.pdf")))
        if not pdf_files:
            logger.error(f"No PDF files found in folder: {input_path}")
            sys.exit(1)

        # Group output under output/{folder_stem}/
        group_output_dir = out_dir / input_path.stem
        group_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(pdf_files)} PDFs from folder: {input_path}")
        logger.info(f"Grouped output folder: {group_output_dir}")

        for pdf in pdf_files:
            process_single_pdf(pdf, group_output_dir)
    else:
        logger.error(f"Unsupported path type: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()