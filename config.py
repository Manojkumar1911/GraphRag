from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

try:  # noqa: SIM105 - optional dependency
    from dotenv import load_dotenv

    load_dotenv()  # type: ignore[func-returns-value]
except Exception:  # pragma: no cover - silently proceed if dotenv unavailable
    pass


@dataclass(slots=True)
class GraphRAGConfig:
    """Configuration container for the Graph RAG pipeline."""

    kb_path: Path = Path(os.getenv("KB_PATH", "kb.txt")).resolve()
    chroma_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_store")).resolve()
    faiss_dir: Path = Path(os.getenv("FAISS_PERSIST_DIR", "faiss_store")).resolve()

    neo4j_uri: str | None = os.getenv("NEO4J_URI")
    neo4j_username: str | None = os.getenv("NEO4J_USERNAME")
    neo4j_password: str | None = os.getenv("NEO4J_PASSWORD")
    neo4j_database: str | None = os.getenv("NEO4J_DATABASE", "neo4j")

    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    chunk_similarity_threshold: float = float(os.getenv("CHUNK_SIMILARITY_THRESHOLD", "0.8"))
    entity_merge_similarity_threshold: float = float(os.getenv("ENTITY_MERGE_SIMILARITY_THRESHOLD", "0.85"))
    max_sentences_per_chunk: int = int(os.getenv("MAX_SENTENCES_PER_CHUNK", "4"))
    louvain_resolution: float = float(os.getenv("LOUVAIN_RESOLUTION", "1.0"))
    top_k_communities: int = int(os.getenv("GRAPH_RAG_TOP_K", "3"))

    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "graph_rag_communities")

    traditional_chunk_size_tokens: int = int(os.getenv("TRAD_CHUNK_SIZE_TOKENS", "350"))
    traditional_chunk_overlap_tokens: int = int(os.getenv("TRAD_CHUNK_OVERLAP_TOKENS", "75"))
    traditional_top_k: int = int(os.getenv("TRAD_TOP_K", "4"))

    def validate(self) -> None:
        missing = []
        if not self.kb_path.exists():
            missing.append(f"Knowledge base file not found: {self.kb_path}")
        if not self.neo4j_uri:
            missing.append("NEO4J_URI is required")
        if not self.neo4j_username:
            missing.append("NEO4J_USERNAME is required")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD is required")
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY is required")

        if missing:
            raise ValueError("\n".join(missing))

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
