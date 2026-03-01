"""
VisualizeGraphE - Graph Visualization for Plan E
Visualizes the document graph structure created by vectorizerE.py
Supports multiple visualization methods: interactive HTML, static image, and network analysis
"""

import json
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("ERROR: networkx is required. Install with: pip install networkx")
    sys.exit(1)

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("WARNING: pyvis not available for interactive visualization. Install with: pip install pyvis")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- GRAPH LOADING ---------------- 
def load_graph(graph_file: Path) -> nx.DiGraph:
    """Load graph from JSON file"""
    if not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        sys.exit(1)
    
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node_data in graph_data.get("nodes", []):
        node_id = node_data.pop("id")
        G.add_node(node_id, **node_data)
    
    # Add edges
    for edge_data in graph_data.get("edges", []):
        source = edge_data.pop("source")
        target = edge_data.pop("target")
        G.add_edge(source, target, **edge_data)
    
    logger.info(f"Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

# ---------------- STATIC VISUALIZATION (Matplotlib) ---------------- 
def visualize_static(G: nx.DiGraph, output_file: Path, layout: str = "hierarchical"):
    """Create static visualization using matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available for static visualization")
        return
    
    logger.info(f"Creating static visualization with {layout} layout...")
    
    # Choose layout
    if layout == "hierarchical":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Separate nodes by type
    section_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "section"]
    chunk_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "chunk"]
    page_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "page"]
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Separate edges by type for different styling
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) 
                     if d.get("relation") != "similar_to"]
    similarity_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d.get("relation") == "similar_to"]
    
    # Draw regular edges
    if regular_edges:
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                              alpha=0.3, arrows=True, arrowsize=10, 
                              edge_color='gray', width=0.5)
    
    # Draw similarity edges with different color and style
    if similarity_edges:
        nx.draw_networkx_edges(G, pos, edgelist=similarity_edges,
                              alpha=0.6, arrows=True, arrowsize=12,
                              edge_color='purple', width=1.5, style='dashed')
    
    # Draw nodes by type with different colors
    if section_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=section_nodes, 
                              node_color='lightblue', node_size=500, alpha=0.8, label='Sections')
        # Add labels for sections
        section_labels = {n: G.nodes[n].get("section_title", n.split(":")[-1])[:30] 
                         for n in section_nodes}
        nx.draw_networkx_labels(G, pos, section_labels, font_size=6, font_weight='bold')
    
    if chunk_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=chunk_nodes, 
                              node_color='lightgreen', node_size=100, alpha=0.6, label='Chunks')
    
    if page_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=page_nodes, 
                              node_color='lightcoral', node_size=300, alpha=0.7, label='Pages')
    
    # Count similarity edges
    similarity_count = len([e for e in G.edges(data=True) if e[2].get("relation") == "similar_to"])
    
    plt.title(f"Document Graph Visualization\n{len(G.nodes)} nodes, {len(G.edges)} edges ({similarity_count} similarity edges)", 
              fontsize=16, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=10, label='Sections', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
               markersize=6, label='Chunks', alpha=0.6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
               markersize=8, label='Pages', alpha=0.7),
        Line2D([0], [0], color='gray', linewidth=1, label='Regular edges', alpha=0.3),
        Line2D([0], [0], color='purple', linewidth=1.5, linestyle='--', 
               label='Similarity edges', alpha=0.6)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Static visualization saved to: {output_file}")
    plt.close()

# ---------------- INTERACTIVE VISUALIZATION (Pyvis) ---------------- 
def visualize_interactive(G: nx.DiGraph, output_file: Path, height: str = "800px", width: str = "100%"):
    """Create interactive HTML visualization using pyvis"""
    if not PYVIS_AVAILABLE:
        logger.error("pyvis not available for interactive visualization")
        return
    
    logger.info("Creating interactive visualization...")
    
    # Create network
    net = Network(height=height, width=width, directed=True, bgcolor="#222222", font_color="white")
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 12
        },
        "scaling": {
          "min": 10,
          "max": 50
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true
          }
        },
        "smooth": {
          "type": "continuous"
        }
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion"
      }
    }
    """)
    
    # Add nodes with styling
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get("type", "unknown")
        
        # Default values
        color = "#BD10E0"  # Purple
        size = 20
        title = str(node_id)
        label = node_id.split(":")[-1]
        
        # Sections: use clean section title (NOT full section_path/id)
        if node_type == "section":
            color = "#4A90E2"  # Blue
            size = 30
            section_title = node_data.get("section_title") or ""
            section_path = node_data.get("section_path") or ""
            # Label: prefer explicit section_title, otherwise last part of section_path
            if section_title:
                label = section_title
            elif section_path:
                label = section_path.split(" > ")[-1]
            else:
                label = label  # fallback: id tail
            title = f"Section: {section_title or label}"
        
        # Chunks: show chunk id and heading
        elif node_type == "chunk":
            color = "#7ED321"  # Green
            size = 15
            chunk_id = node_data.get("chunk_id", "?")
            heading = node_data.get("heading", "No heading")
            label = f"{chunk_id}"
            title = f"Chunk {chunk_id}: {heading[:80]}"
        
        # Pages: show page number
        elif node_type == "page":
            color = "#F5A623"  # Orange
            size = 25
            page_num = node_data.get("page_number", "?")
            label = f"Page {page_num}"
            title = f"Page {page_num}"
        
        # Truncate label to keep visualization readable
        label = str(label)[:30]
        
        # Add node
        net.add_node(
            node_id,
            label=label,
            color=color,
            size=size,
            title=title,
            group=node_type,
        )
    
    # Add edges with relation labels and styling
    for source, target, edge_data in G.edges(data=True):
        relation = edge_data.get("relation", "unknown")
        
        # Style similarity edges differently
        if relation == "similar_to":
            similarity = edge_data.get("similarity", 0.0)
            color = "#9B59B6"  # Purple
            width = 3
            title = f"Similar (similarity: {similarity:.2f})"
        elif relation == "follows":
            color = "#3498DB"  # Blue
            width = 2
            title = relation
        elif relation in ["contains", "belongs_to"]:
            color = "#2ECC71"  # Green
            width = 2
            title = relation
        elif relation == "on_page":
            color = "#E67E22"  # Orange
            width = 2
            title = relation
        else:
            color = "#888888"  # Gray
            width = 1
            title = relation
        
        net.add_edge(source, target, title=title, color=color, width=width)
    
    # Save
    net.save_graph(str(output_file))
    logger.info(f"Interactive visualization saved to: {output_file}")
    logger.info(f"Open {output_file} in a web browser to view the interactive graph")

# ---------------- SIMPLIFIED VISUALIZATION (Subgraph) ---------------- 
def visualize_simplified(G: nx.DiGraph, output_file: Path, max_nodes: int = 100):
    """Create simplified visualization showing only sections and sample chunks"""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return
    
    logger.info(f"Creating simplified visualization (max {max_nodes} nodes)...")
    
    # Create subgraph with sections and some chunks
    section_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "section"]
    chunk_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "chunk"][:max_nodes - len(section_nodes)]
    
    subgraph_nodes = section_nodes + chunk_nodes
    subgraph = G.subgraph(subgraph_nodes)
    
    # Layout
    pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
    
    plt.figure(figsize=(16, 12))
    
    # Separate edges by type
    regular_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                     if d.get("relation") != "similar_to"]
    similarity_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                        if d.get("relation") == "similar_to"]
    
    # Draw regular edges
    if regular_edges:
        nx.draw_networkx_edges(subgraph, pos, edgelist=regular_edges,
                              alpha=0.2, arrows=True, arrowsize=8, 
                              edge_color='gray', width=0.3)
    
    # Draw similarity edges
    if similarity_edges:
        nx.draw_networkx_edges(subgraph, pos, edgelist=similarity_edges,
                              alpha=0.5, arrows=True, arrowsize=10,
                              edge_color='purple', width=1.0, style='dashed')
    
    # Draw section nodes
    sub_sections = [n for n in subgraph.nodes() if subgraph.nodes[n].get("type") == "section"]
    if sub_sections:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=sub_sections,
                              node_color='lightblue', node_size=800, alpha=0.9)
        section_labels = {n: G.nodes[n].get("section_title", n.split(":")[-1])[:25] 
                         for n in sub_sections}
        nx.draw_networkx_labels(subgraph, pos, section_labels, font_size=8, font_weight='bold')
    
    # Draw chunk nodes
    sub_chunks = [n for n in subgraph.nodes() if subgraph.nodes[n].get("type") == "chunk"]
    if sub_chunks:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=sub_chunks,
                              node_color='lightgreen', node_size=50, alpha=0.5)
    
    similarity_count = len([e for e in subgraph.edges(data=True) if e[2].get("relation") == "similar_to"])
    plt.title(f"Simplified Document Graph\n{len(subgraph.nodes)} nodes shown (of {len(G.nodes)} total), {similarity_count} similarity edges", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Simplified visualization saved to: {output_file}")
    plt.close()

# ---------------- GRAPH STATISTICS ---------------- 
def print_graph_stats(G: nx.DiGraph):
    """Print statistics about the graph"""
    print("\n" + "=" * 80)
    print("GRAPH STATISTICS")
    print("=" * 80)
    
    # Node counts by type
    node_types = {}
    for node_id in G.nodes():
        node_type = G.nodes[node_id].get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\nTotal Nodes: {len(G.nodes)}")
    for node_type, count in sorted(node_types.items()):
        print(f"  - {node_type.capitalize()}: {count}")
    
    # Edge counts by relation
    edge_relations = {}
    for source, target, edge_data in G.edges(data=True):
        relation = edge_data.get("relation", "unknown")
        edge_relations[relation] = edge_relations.get(relation, 0) + 1
    
    print(f"\nTotal Edges: {len(G.edges)}")
    for relation, count in sorted(edge_relations.items()):
        if relation == "similar_to":
            print(f"  - {relation}: {count} (similarity connections)")
        else:
            print(f"  - {relation}: {count}")
    
    # Show similarity edge statistics
    if "similar_to" in edge_relations:
        similarity_edges = [(u, v, d) for u, v, d in G.edges(data=True) 
                           if d.get("relation") == "similar_to"]
        if similarity_edges:
            similarities = [d.get("similarity", 0.0) for u, v, d in similarity_edges if d.get("similarity")]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                min_sim = min(similarities)
                max_sim = max(similarities)
                print(f"\nSimilarity Edge Statistics:")
                print(f"  - Average similarity: {avg_similarity:.3f}")
                print(f"  - Min similarity: {min_sim:.3f}")
                print(f"  - Max similarity: {max_sim:.3f}")
    
    # Graph metrics
    if len(G.nodes) > 0:
        print(f"\nGraph Metrics:")
        print(f"  - Density: {nx.density(G):.4f}")
        
        # Find sections
        section_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "section"]
        if section_nodes:
            print(f"  - Number of sections: {len(section_nodes)}")
        
        # Average chunks per section
        chunk_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "chunk"]
        if section_nodes and chunk_nodes:
            avg_chunks = len(chunk_nodes) / len(section_nodes) if section_nodes else 0
            print(f"  - Average chunks per section: {avg_chunks:.1f}")
    
    print("=" * 80 + "\n")

# ---------------- MAIN FUNCTION ---------------- 
def find_graph_file(directory: Path) -> Optional[Path]:
    """Find document graph JSON file in directory"""
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Visualize document graph from vectorizerE.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use folder name (auto-detects E subfolder)
  python visualizeGraphE.py output/HDFC-Life-Cancer-Care-101N106V04-Policy-Document
  
  # Use most recent folder in output directory
  python visualizeGraphE.py
  
  # Specify graph file directly
  python visualizeGraphE.py --graph-file path/to/graph.json
        """
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to folder containing document (will look for E subfolder), or use most recent folder if not specified"
    )
    parser.add_argument(
        "--graph-file",
        type=str,
        help="Explicit path to graph JSON file (overrides auto-detection)"
    )
    
    args = parser.parse_args()
    
    # Determine graph file path
    if args.graph_file:
        graph_file = Path(args.graph_file)
        base_path = graph_file.parent
    elif args.path:
        input_path = Path(args.path)
        
        if input_path.is_file() and input_path.suffix == ".json":
            # User provided a JSON file directly
            graph_file = input_path
            base_path = input_path.parent
        elif input_path.is_dir():
            # User provided a folder, look for E subfolder
            plan_e_dir = input_path / "E"
            if plan_e_dir.exists():
                base_path = plan_e_dir
                graph_file = find_graph_file(plan_e_dir)
            else:
                # Try the folder itself
                base_path = input_path
                graph_file = find_graph_file(input_path)
            
            if not graph_file:
                logger.error(f"No graph file found in: {input_path}")
                logger.error("Looking for files matching pattern: *_document_graph.json")
                logger.error("Please run vectorizerE.py first to generate the graph")
                sys.exit(1)
        else:
            logger.error(f"Invalid path: {input_path}")
            sys.exit(1)
    else:
        # Default: look for most recent folder in output directory
        output_dir = Path("output")
        if output_dir.exists():
            folders = [d for d in output_dir.iterdir() if d.is_dir()]
            if folders:
                input_folder = max(folders, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent folder: {input_folder}")
                plan_e_dir = input_folder / "E"
                if plan_e_dir.exists():
                    base_path = plan_e_dir
                    graph_file = find_graph_file(plan_e_dir)
                else:
                    logger.error(f"Plan E folder not found: {plan_e_dir}")
                    sys.exit(1)
            else:
                logger.error("No folders found in output directory. Please specify a folder path.")
                sys.exit(1)
        else:
            logger.error("Output directory not found. Please specify a folder path.")
            sys.exit(1)
    
    if not graph_file or not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Graph Visualization - Plan E")
    logger.info("=" * 80)
    logger.info(f"Loading graph from: {graph_file}")
    logger.info(f"Base directory: {base_path}")
    
    # Load graph
    G = load_graph(graph_file)
    
    # Print statistics
    print_graph_stats(G)
    
    # Create output directory
    output_dir = base_path / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Use graph file stem for output file names
    graph_name = graph_file.stem.replace("_document_graph", "")
    
    # 1. Interactive HTML visualization (best for exploration)
    if PYVIS_AVAILABLE:
        interactive_file = output_dir / f"{graph_name}_interactive.html"
        logger.info("\nCreating interactive visualization...")
        visualize_interactive(G, interactive_file)
        print(f"\n✓ Interactive visualization: {interactive_file}")
        print("  Open this file in a web browser for an interactive graph!")
    
    # 2. Simplified static visualization (sections + sample chunks)
    if MATPLOTLIB_AVAILABLE:
        simplified_file = output_dir / f"{graph_name}_simplified.png"
        logger.info("\nCreating simplified visualization...")
        visualize_simplified(G, simplified_file, max_nodes=150)
        print(f"✓ Simplified visualization: {simplified_file}")
        
        # 3. Full static visualization (if graph is not too large)
        if len(G.nodes) < 500:
            static_file = output_dir / f"{graph_name}_full.png"
            logger.info("\nCreating full static visualization...")
            visualize_static(G, static_file, layout="spring")
            print(f"✓ Full static visualization: {static_file}")
        else:
            logger.info(f"\nSkipping full visualization (graph too large: {len(G.nodes)} nodes)")
            logger.info("  Use simplified or interactive visualization instead")
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    
    if not PYVIS_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        print("\nWARNING: No visualization libraries available!")
        print("Install one of:")
        print("  - pyvis: pip install pyvis (for interactive HTML)")
        print("  - matplotlib: pip install matplotlib (for static images)")

if __name__ == "__main__":
    main()
