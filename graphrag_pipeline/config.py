"""
Shim configuration module for graphrag_pipeline.
It reuses the root GraphRAGConfig so there is still a single source of truth
for all environment variables, while exposing the constants and heavyweight
model objects expected by the pipeline modules.
"""

from __future__ import annotations

import os
import spacy
from sentence_transformers import SentenceTransformer

from config import GraphRAGConfig

# Instantiate the root configuration once so we reuse shared settings everywhere
CONFIG = GraphRAGConfig()

# Neo4j configuration (mirrors root config values)
NEO4J_URI = CONFIG.neo4j_uri
NEO4J_USERNAME = CONFIG.neo4j_username
NEO4J_PASSWORD = CONFIG.neo4j_password
NEO4J_DATABASE = CONFIG.neo4j_database or "neo4j"

# Groq LLM configuration
GROQ_API_KEY = CONFIG.groq_api_key
GROQ_MODEL = CONFIG.groq_model
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_BASE_BACKOFF_SECONDS = float(os.getenv("GROQ_BASE_BACKOFF_SECONDS", "2"))

# Vector database / persistence paths
CHROMA_PERSIST_DIR = str(CONFIG.chroma_dir)
INDEX_DIR = os.getenv("INDEX_DIR", "./graphrag_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.json")

# spaCy / NLP configuration
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        f"spaCy model '{SPACY_MODEL}' is not available. "
        f"Install it via 'python -m spacy download {SPACY_MODEL}'."
    ) from exc

# SentenceTransformer embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    """Generate embedding for text using SentenceTransformer."""
    if not text or not isinstance(text, str):
        return None
    try:
        return embedder.encode(text, convert_to_tensor=False)
    except Exception as exc:  # pragma: no cover - upstream runtime errors
        print(f"Error generating embedding: {exc}")
        return None


# Graph algorithm configuration (with env overrides for advanced tuning)
GRAPH_PROJECTION_NAME = os.getenv("GRAPH_PROJECTION_NAME", "entityGraph")
COMMUNITY_PROPERTY = os.getenv("COMMUNITY_PROPERTY", "communityId")
WCC_PROPERTY = os.getenv("WCC_PROPERTY", "wccId")
PAGERANK_PROPERTY = os.getenv("PAGERANK_PROPERTY", "pagerank")
COMMUNITY_SEED_LIMIT = int(os.getenv("COMMUNITY_SEED_LIMIT", "5"))
SUMMARY_TOP_N = int(os.getenv("SUMMARY_TOP_N", "50"))
SUPPORTING_TEXT_LIMIT = int(os.getenv("SUPPORTING_TEXT_LIMIT", "5"))
