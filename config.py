from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List

_ENV_INITIALIZED = False


def _manual_env_load(env_path: Path) -> None:
    """Fallback parser for .env files when python-dotenv is unavailable."""
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not value:
                continue

            if value[0] in {'"', "'"} and value[-1] == value[0]:
                value = value[1:-1]
            else:
                value = value.split("#", 1)[0].strip()

            os.environ.setdefault(key, value)
    except FileNotFoundError:
        return


def _ensure_env_loaded() -> None:
    global _ENV_INITIALIZED
    if _ENV_INITIALIZED:
        return

    env_hint = os.getenv("GRAPH_RAG_ENV_PATH")
    candidates: list[Path] = []
    if env_hint:
        candidates.append(Path(env_hint))

    current_dir = Path(__file__).resolve().parent
    candidates.extend([current_dir / ".env", current_dir.parent / ".env"])

    load_dotenv = None
    try:  # noqa: SIM105
        from dotenv import load_dotenv as _load_dotenv

        load_dotenv = _load_dotenv
    except Exception:  # pragma: no cover
        load_dotenv = None

    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue

        loaded = False
        if load_dotenv:
            try:
                loaded = bool(load_dotenv(dotenv_path=candidate, override=False))
            except Exception:
                loaded = False

        if not loaded:
            _manual_env_load(candidate)

        _ENV_INITIALIZED = True
        return

    _ENV_INITIALIZED = True


_ensure_env_loaded()


@dataclass(slots=True)
class GraphRAGConfig:
    """Configuration container for the Graph RAG pipeline."""

    kb_path: Path = Path(os.getenv("KB_PATH", "kb.txt")).resolve()
    kb_glob: str = os.getenv("KB_GLOB", "*.txt")
    kb_paths: List[Path] = field(default_factory=list)
    chroma_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_store")).resolve()
    faiss_dir: Path = Path(os.getenv("FAISS_PERSIST_DIR", "faiss_store")).resolve()

    neo4j_uri: str | None = os.getenv("NEO4J_URI")
    neo4j_username: str | None = os.getenv("NEO4J_USERNAME")
    neo4j_password: str | None = os.getenv("NEO4J_PASSWORD")
    neo4j_database: str | None = os.getenv("NEO4J_DATABASE", "neo4j")

    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
        self.refresh_kb_paths()

        if not self.kb_paths:
            missing.append(
                "No knowledge base files found. Set KB_PATH, KB_GLOB, or place TXT files in the project root."
            )
        if not self.neo4j_uri:
            missing.append("NEO4J_URI is required")
        if not self.neo4j_username:
            missing.append("NEO4J_USERNAME is required")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD is required")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY is required")

        if missing:
            raise ValueError("\n".join(missing))

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def refresh_kb_paths(self) -> List[Path]:
        """Discover KB files based on KB_PATH / KB_GLOB settings."""
        kb_dir = self.kb_path.parent
        discovered: list[Path] = []

        if kb_dir.exists():
            discovered.extend(sorted(kb_dir.glob(self.kb_glob)))

        if self.kb_path.exists() and self.kb_path not in discovered:
            discovered.append(self.kb_path)

        self.kb_paths = [path.resolve() for path in discovered if path.is_file()]
        return self.kb_paths

    def get_all_kb_text(self) -> str:
        """Read and concatenate text from all discovered KB files."""
        paths = self.refresh_kb_paths()

        if not paths:
            raise FileNotFoundError(
                "No knowledge base files available. Provide KB_PATH/KB_GLOB or place TXT files in the repository."
            )

        texts: list[str] = []
        missing: list[str] = []

        for path in paths:
            try:
                content = path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                missing.append(str(path))
                continue

            if content:
                texts.append(content)

        if missing:
            print(f"⚠️  Skipped missing knowledge base files: {', '.join(missing)}")

        if not texts:
            raise ValueError("Knowledge base files are empty. Add content before rebuilding the graph.")

        return "\n\n".join(texts)
