"""
GraphRAG Pipeline Package
Clean, modular implementation of Graph-based RAG
"""

from .config import *
from .database import verify_neo4j_connection, clear_graph, _close_neo4j_driver
from .graph_operations import (
    semantic_chunk, extract_entities_relations, 
    detect_communities, summarize_community
)
from .retrieval import retrieve_topk, graph_reasoning_multi_hop, fuse_context
from .llm import llm_call, generate_answer
from .pipeline import build_index, query_graphrag, graph_rag_pipeline, index_exists
from .utils import visualize_knowledge_graph

__version__ = "1.0.0"

# Main API
__all__ = [
    # Primary functions
    "build_index",
    "query_graphrag",
    "graph_rag_pipeline",
    
    # Utilities
    "visualize_knowledge_graph",
    "verify_neo4j_connection",
    "clear_graph",
    "index_exists",
    
    # Advanced
    "semantic_chunk",
    "extract_entities_relations",
    "detect_communities",
    "retrieve_topk",
    "graph_reasoning_multi_hop",
    "generate_answer",
]
