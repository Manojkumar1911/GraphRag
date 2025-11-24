"""
Interactive Graph RAG CLI
=========================

This CLI orchestrates both graph-based and traditional RAG pipelines:

Graph RAG:
    • Semantic chunking over TXT knowledge bases
    • Entity extraction using transformer-based NER
    • Neo4j for persistent knowledge graphs
    • Community detection
    • Groq LLM summarization and answer synthesis
    • ChromaDB for vector search on community summaries

Traditional RAG:
    • Mechanical chunking
    • FAISS vector search
    • Direct context retrieval

Usage:
    python interactive_rag_pipeline.py
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

# Try importing visualization (optional)
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  matplotlib/networkx not available. Visualization disabled.")

# Import from new organized graphrag_pipeline package
from graphrag_pipeline import (
    build_index,
    query_graphrag,
    index_exists,
    _close_neo4j_driver,
)
from graphrag_pipeline.database import (
    vector_db,
    _with_session,
)
from graphrag_pipeline.config import NEO4J_DATABASE

# Import other dependencies (keep these as-is)
from config import GraphRAGConfig
from traditional_pipeline import TraditionalRAGPipeline


class GraphRAGCLI:
    """Interactive wrapper around both GraphRAG and Traditional RAG pipelines."""

    def __init__(self) -> None:
        self.config = GraphRAGConfig()
        self.config.refresh_kb_paths()
        
        # Initialize pipelines
        print("Initializing Graph RAG pipeline...")
        self.graph_config = self.config
        
        print("Initializing Traditional RAG pipeline...")
        self.trad_pipeline = TraditionalRAGPipeline(self.config)
        
        # Check readiness
        self._graph_ready = self._check_graph_index()
        self._trad_ready = self.trad_pipeline.has_index()

    def _check_graph_index(self) -> bool:
        """Check if graph index is ready."""
        if not index_exists():
            return False

        try:
            def _has_nodes(session):
                record = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()
                return bool(record and record.get("count", 0) > 0)

            return _with_session(_has_nodes)
        except Exception as e:
            logging.debug(f"Graph index check failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Build & maintenance
    # ------------------------------------------------------------------
    def build_graph(self, full_reset: bool = False) -> None:
        """Build Graph RAG pipeline."""
        print("\n🛠️  Building Graph RAG pipeline...\n")
        
        try:
            kb_files = self.config.refresh_kb_paths()
            if not kb_files:
                print("❌ No knowledge base files available. Place TXT/PDF files (subfolders are scanned) or update KB_PATH/KB_GLOB.")
                return

            print(
                "📚 Using knowledge base files: "
                + ", ".join(path.name for path in kb_files)
            )

            text = self.graph_config.get_all_kb_text()
            build_index(text, clear_db=full_reset)
            
            print("✅ Graph RAG pipeline built successfully.")
            self._graph_ready = True
            
        except Exception as e:
            print(f"❌ Failed to build graph pipeline: {e}")
            logging.exception("Graph build failed", exc_info=e)

    def build_trad(self, reset_index: bool = True) -> None:
        """Build Traditional RAG FAISS index."""
        print("\n🛠️  Building traditional RAG FAISS index...\n")
        
        try:
            chunks = self.trad_pipeline.build(reset_index=reset_index)
            print(f"✅ Indexed {len(chunks)} chunks. Total: {self.trad_pipeline.chunk_count()}.")
            self._trad_ready = True
        except Exception as e:
            print(f"❌ Failed to build traditional pipeline: {e}")
            logging.exception("Traditional build failed", exc_info=e)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query_graph(self, question: str) -> str:
        """Query Graph RAG system."""
        if not self._graph_ready:
            return "❌ Graph RAG pipeline not built. Run 'build graph' first."
        
        try:
            kb_files = self.config.refresh_kb_paths()
            if not kb_files:
                return "❌ No knowledge base files configured."

            if not index_exists():
                print("\nℹ️  No GraphRAG index found. Building once before querying...\n")
                text = self.config.get_all_kb_text()
                build_index(text, clear_db=True)
                self._graph_ready = True

            answer = query_graphrag(question)
            if not answer:
                return "⚠️  Graph RAG returned no answer."

            return answer

        except Exception as exc:
            print(f"❌ Graph RAG query failed: {exc}")
            logging.exception("Graph RAG query failed", exc_info=exc)
            return f"❌ Error: {str(exc)}"

    def query_trad(self, question: str) -> None:
        """Query Traditional RAG system."""
        if not self._trad_ready:
            print("⚠️  Traditional RAG index is empty. Run 'build trad' first.")
            return

        print("\n" + "=" * 80)
        print(f"🔍 TRADITIONAL RAG QUESTION: {question}")
        print("=" * 80)

        try:
            result = self.trad_pipeline.query(question)
            answer = result.get("answer", "(no answer)")

            print("\n💬 TRADITIONAL ANSWER\n" + "-" * 80)
            print(textwrap.fill(answer, width=80))

        except Exception as exc:
            print(f"❌ Traditional RAG query failed: {exc}")
            logging.exception("Traditional RAG query failed", exc_info=exc)

        print("\n" + "=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def visualize_graph(self) -> None:
        """Visualize the Neo4j knowledge graph."""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️  Visualization requires matplotlib and networkx.")
            print("   Install with: pip install matplotlib networkx")
            return
        
        try:
            def _fetch_edges(session):
                query = session.run(
                    """
                    MATCH (n:Entity)-[r:RELATION]-(m:Entity)
                    RETURN n.name AS source, m.name AS target, r.type AS rel_type
                    LIMIT 100
                    """
                )
                return [(record["source"], record["target"]) for record in query]

            edges = _with_session(_fetch_edges)

            if not edges:
                print("⚠️  Graph is empty. Build the pipeline first.")
                return
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_edges_from(edges)
            
            print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Visualize
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
            
            nx.draw_networkx_nodes(
                G, pos,
                node_color='lightblue',
                node_size=1200,
                alpha=0.9
            )
            nx.draw_networkx_edges(
                G, pos,
                width=2,
                alpha=0.3,
                edge_color='gray'
            )
            nx.draw_networkx_labels(
                G, pos,
                font_size=8,
                font_weight='bold'
            )
            
            plt.title("Graph RAG Knowledge Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            logging.exception("Visualization error", exc_info=e)

    def stats(self) -> None:
        """Display statistics for both pipelines."""
        print("\n📈 PIPELINE STATS")
        print("-" * 80)
        
        # Graph stats
        try:
            def _counts(session):
                node_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
                edge_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS count").single()["count"]
                return node_count, edge_count

            node_count, edge_count = _with_session(_counts)
            community_count = vector_db.count()
            
            print(f"Graph RAG:")
            print(f"  • Nodes (entities): {node_count}")
            print(f"  • Edges (relationships): {edge_count}")
            print(f"  • Communities indexed: {community_count}")
        except Exception as e:
            print(f"Graph RAG: ❌ Error - {e}")
        
        # Traditional stats
        try:
            chunk_count = self.trad_pipeline.chunk_count()
            print(f"\nTraditional RAG:")
            print(f"  • Chunks indexed: {chunk_count}")
        except Exception as e:
            print(f"Traditional RAG: ❌ Error - {e}")
        
        print("-" * 80)

    def close(self) -> None:
        """Close all pipeline connections."""
        try:
            self.trad_pipeline.close()
        except:
            pass

        try:
            _close_neo4j_driver()
        except Exception:
            pass


# ============================================================================
# INTERACTIVE CLI INTERFACE
# ============================================================================

def print_banner() -> None:
    """Print welcome banner."""
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        🕸️  GRAPH RAG INTERACTIVE DEMO                                      ║
║                                                                            ║
║   Compare Graph RAG vs Traditional RAG side-by-side!                       ║
║   Powered by Neo4j, ChromaDB, FAISS, and Groq LLM.                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu() -> None:
    """Print command menu."""
    menu = """
┌─────────────────────────────────────────────────────────────────────────┐
│ COMMANDS                                                                │
│   build graph   → Run graph pipeline (chunk → entities → Neo4j → Chroma)│
│   rebuild graph → Full graph reset (Neo4j + Chroma)                     │
│   build trad    → Build traditional FAISS index                         │
│   rebuild trad  → Reset and rebuild FAISS index                         │
│   stats         → Show both pipeline statistics                         │
│   viz           → Visualize Neo4j knowledge graph                       │
│   help          → Show this menu                                        │
│   quit          → Exit program                                          │
│                                                                         │
│ QUERIES                                                                 │
│   <question>    → Query BOTH pipelines                                  │
│   graph: <q>    → Query only Graph RAG                                  │
│   trad: <q>     → Query only Traditional RAG                            │
└─────────────────────────────────────────────────────────────────────────┘
    """
    print(menu)


def show_examples() -> None:
    """Show example queries."""
    examples = """
📝 EXAMPLE QUERIES:

1. "What companies did Elon Musk found?"
2. "Summarize SpaceX achievements."
3. "How is Tesla expanding?"
4. "What is Neuralink working on?"
5. "Explain Mars colonization plans."

💡 TIP: Try the same question on both systems to compare!
    """
    print(examples)


def main():
    """Main interactive loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    print_banner()

    # Initialize CLI
    try:
        cli = GraphRAGCLI()
    except Exception as exc:
        print(f"\n❌ Failed to initialize pipelines: {exc}")
        logging.exception("CLI initialization failed", exc_info=exc)
        print("\n💡 Check your .env file and ensure all dependencies are installed.")
        sys.exit(1)

    # Check readiness
    if not cli._graph_ready:
        print(
            "\nℹ️  No graph index detected. Run 'build graph' to create one "
            "from kb.txt before querying."
        )
    if not cli._trad_ready:
        print(
            "ℹ️  No traditional index detected. Run 'build trad' to create one."
        )

    print_menu()

    # Main loop
    try:
        while True:
            user_input = input("\n🔍 Command or question: ").strip()
            
            if not user_input:
                continue

            lowered = user_input.lower()
            
            # Exit commands
            if lowered in {"quit", "exit", "q"}:
                print("\n👋 Thanks for exploring Graph RAG! Goodbye.\n")
                break
            
            # Help commands
            if lowered in {"help", "h", "?"}:
                print_menu()
                continue
            
            if lowered in {"examples", "ex"}:
                show_examples()
                continue
            
            # Build commands
            if lowered == "build graph":
                cli.build_graph(full_reset=False)
                continue
            
            if lowered == "rebuild graph":
                cli.build_graph(full_reset=True)
                continue
            
            if lowered == "build trad":
                cli.build_trad(reset_index=False)
                continue
            
            if lowered == "rebuild trad":
                cli.build_trad(reset_index=True)
                continue
            
            # Info commands
            if lowered in {"stats", "status", "info"}:
                cli.stats()
                continue
            
            if lowered in {"viz", "graph", "visualize", "show"}:
                cli.visualize_graph()
                continue
            
            # Query commands
            if lowered.startswith("graph:"):
                question = user_input.split(":", 1)[1].strip()
                if question:
                    answer = cli.query_graph(question)
                    if answer:
                        print("\n💬 GRAPH RAG ANSWER\n" + "-" * 80)
                        print(textwrap.fill(answer, width=80))
                        print("\n" + "=" * 80 + "\n")
                else:
                    print("⚠️  Provide a question after 'graph:'")
                continue
            
            if lowered.startswith("trad:"):
                question = user_input.split(":", 1)[1].strip()
                if question:
                    cli.query_trad(question)
                else:
                    print("⚠️  Provide a question after 'trad:'")
                continue
            
            # Default: Traditional first, Graph on demand
            print("\n🔄 Querying Traditional RAG...\n")
            cli.query_trad(user_input)

            prompt = (
                "\n↩️  Press Enter to fetch the Graph RAG response, or type 'skip' to continue: "
            )
            follow_up = input(prompt).strip().lower()

            if follow_up in {"", "y", "yes"}:
                print("\n🕸️  Querying Graph RAG...\n")
                graph_answer = cli.query_graph(user_input)
                if graph_answer:
                    print("\n💬 GRAPH RAG ANSWER\n" + "-" * 80)
                    print(textwrap.fill(graph_answer, width=80))
                    print("\n" + "=" * 80 + "\n")
                else:
                    print("⚠️  No Graph RAG answer returned.")
            else:
                print("\n⏭️  Skipping Graph RAG response.\n")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    finally:
        cli.close()


if __name__ == "__main__":
    main()
