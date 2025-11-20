"""
Database Management
- Neo4j driver with connection pooling and retry logic
- ChromaDB vector store initialization
"""

import time
from threading import Lock
from typing import Callable, TypeVar
from neo4j import GraphDatabase, Session, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired
import chromadb

# Import from ROOT config.py (not from .config)
from config import GraphRAGConfig

# Initialize config instance
CONFIG = GraphRAGConfig()

# ============================================================
# NEO4J DRIVER MANAGEMENT
# ============================================================

_neo4j_driver: Driver | None = None
_neo4j_lock = Lock()


def _get_neo4j_driver():
    """Get or create Neo4j driver (singleton pattern)"""
    global _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver is None:
            if not CONFIG.neo4j_uri:
                raise RuntimeError(
                    "NEO4J_URI is not set. Please specify a Bolt or Neo4j URI in your environment, "
                    "e.g. bolt://localhost:7687"
                )
            if not CONFIG.neo4j_username or not CONFIG.neo4j_password:
                raise RuntimeError(
                    "NEO4J_USERNAME and NEO4J_PASSWORD must be set in the environment."
                )
            _neo4j_driver = GraphDatabase.driver(
                CONFIG.neo4j_uri,
                auth=(CONFIG.neo4j_username, CONFIG.neo4j_password)
            )
        return _neo4j_driver


def _close_neo4j_driver():
    """Close Neo4j driver connection"""
    global _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver:
            _neo4j_driver.close()
            _neo4j_driver = None


T = TypeVar("T")


def _with_session(func: Callable[[Session], T], *, max_attempts: int = 3) -> T:
    """Execute function with Neo4j session, with retry logic for connection issues"""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        driver = _get_neo4j_driver()
        try:
            with driver.session(database=CONFIG.neo4j_database) as session:
                return func(session)
        except (ServiceUnavailable, SessionExpired) as exc:
            last_error = exc
            print(
                f"Neo4j session error (attempt {attempt}/{max_attempts}): {exc}. "
                "Reinitializing driver..."
            )
            _close_neo4j_driver()
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            raise exc
    
    if last_error:
        raise last_error
    raise RuntimeError("Neo4j operation failed without specific error")


def verify_neo4j_connection() -> dict:
    """Verify Neo4j connection and return node/relationship counts"""
    def _get_counts(session):
        node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
        rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()["count"]
        return {"nodes": node_count, "relationships": rel_count}
    
    try:
        counts = _with_session(_get_counts)
        print(f"✅ Neo4j Connected: {counts['nodes']} nodes, {counts['relationships']} relationships")
        return counts
    except Exception as e:
        print(f"❌ Neo4j Connection Error: {e}")
        return {"nodes": 0, "relationships": 0, "error": str(e)}


def clear_graph():
    """Clear all nodes and relationships from Neo4j"""
    def _clear(session: Session):
        session.run("MATCH (n) DETACH DELETE n")
    
    _with_session(_clear)


# ============================================================
# CHROMADB VECTOR STORE
# ============================================================

chroma_client = chromadb.PersistentClient(path=str(CONFIG.chroma_dir))
vector_db = chroma_client.get_or_create_collection(name="rag_communities")