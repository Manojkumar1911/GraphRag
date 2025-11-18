import os
import re
import json
import pickle
import time
from threading import Lock
from typing import Callable, TypeVar
from collections import Counter
from pathlib import Path
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
from neo4j import GraphDatabase, Session, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import spacy
import networkx as nx

try:
    from pyvis.network import Network
except ImportError:
    Network = None

# ---------------------------------------------
# ENV VARIABLES
# ---------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_BASE_BACKOFF_SECONDS = float(os.getenv("GROQ_BASE_BACKOFF_SECONDS", "2"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
INDEX_DIR = os.getenv("INDEX_DIR", "./graphrag_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.json")
GRAPH_PROJECTION_NAME = os.getenv("GRAPH_PROJECTION_NAME", "entityGraph")
COMMUNITY_PROPERTY = os.getenv("COMMUNITY_PROPERTY", "communityId")
WCC_PROPERTY = os.getenv("WCC_PROPERTY", "wccId")
PAGERANK_PROPERTY = os.getenv("PAGERANK_PROPERTY", "pagerank")
COMMUNITY_SEED_LIMIT = int(os.getenv("COMMUNITY_SEED_LIMIT", "5"))
SUMMARY_TOP_N = int(os.getenv("SUMMARY_TOP_N", "50"))
SUPPORTING_TEXT_LIMIT = int(os.getenv("SUPPORTING_TEXT_LIMIT", "5"))

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError as exc:
    raise RuntimeError(
        f"spaCy model '{SPACY_MODEL}' is not available. Install it via 'python -m spacy download {SPACY_MODEL}'."
    ) from exc

# ---------------------------------------------
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


def llm_call(prompt: str) -> str:
    """Call Groq LLM and return text"""
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set. Please configure your Groq credentials.")

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = response.choices[0].message.content if response.choices else ""
            return choice or ""
        except Exception as e:
            err_msg = str(e)
            print(f"LLM call error (attempt {attempt}/{GROQ_MAX_RETRIES}): {err_msg}")
            if attempt == GROQ_MAX_RETRIES:
                break
            backoff = GROQ_BASE_BACKOFF_SECONDS * attempt
            time.sleep(backoff)

    return ""


@lru_cache(maxsize=4)
def _load_prompt_template(filename: str) -> str:
    template_path = Path(__file__).resolve().parent / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()

# ---------------------------------------------
# INITIALIZE Embeddings
# ---------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for the given text using the sentence transformer."""
    if not text or not isinstance(text, str):
        return None
    try:
        # Generate embedding and convert to list for serialization
        return embedder.encode(text, convert_to_tensor=False)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# ---------------------------------------------
# INITIALIZE Vector DB (Chroma)
# ---------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
vector_db = chroma_client.get_or_create_collection(name="rag_communities")

# ---------------------------------------------
# NEO4J DRIVER MANAGEMENT
# ---------------------------------------------
_neo4j_driver: Driver | None = None
_neo4j_lock = Lock()


def _get_neo4j_driver():
    global _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver is None:
            if not NEO4J_URI:
                raise RuntimeError(
                    "NEO4J_URI is not set. Please specify a Bolt or Neo4j URI in your environment, "
                    "e.g. bolt://localhost:7687"
                )
            if not NEO4J_USERNAME or not NEO4J_PASSWORD:
                raise RuntimeError(
                    "NEO4J_USERNAME and NEO4J_PASSWORD must be set in the environment before running the pipeline."
                )
            _neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
        return _neo4j_driver


def _close_neo4j_driver():
    global _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver:
            _neo4j_driver.close()
            _neo4j_driver = None


T = TypeVar("T")


def _with_session(func: Callable[[Session], T], *, max_attempts: int = 3) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        driver = _get_neo4j_driver()
        try:
            with driver.session(database=NEO4J_DATABASE) as session:
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


def attach_supporting_texts(communities, chunks):
    """Populate each community with supporting text snippets from the KB."""
    if not communities or not chunks:
        return

    chunk_cache = [(chunk, chunk.lower()) for chunk in chunks]

    for community in communities:
        members = community.get("members", []) if isinstance(community, dict) else []
        keywords = {name.lower() for name in members if isinstance(name, str)}
        if not keywords:
            community["supporting_texts"] = []
            continue

        scored = []
        for chunk, lowered in chunk_cache:
            count = sum(1 for kw in keywords if kw in lowered)
            if count > 0:
                scored.append((count, chunk.strip()))

        scored.sort(key=lambda item: item[0], reverse=True)
        community["supporting_texts"] = [text for _, text in scored[:SUPPORTING_TEXT_LIMIT]]

def clear_graph():
    """Clear all nodes and relationships from Neo4j"""
    def _clear(session: Session):
        session.run("MATCH (n) DETACH DELETE n")

    _with_session(_clear)

# ---------------------------------------------
# STEP 1: Semantic Chunking
# ---------------------------------------------
def semantic_chunk(text, max_len=300):
    """Split text into semantic chunks"""
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ---------------------------------------------
# STEP 2: Entity & Relationship Extraction (spaCy)
# ---------------------------------------------
def _dedupe_spans(spans):
    unique = []
    seen = set()
    for span in spans:
        key = (span.start, span.end, span.text.lower())
        if key not in seen:
            unique.append(span)
            seen.add(key)
    return unique

def extract_entities_relations(chunk):
    """Extract entities and relationships using spaCy with co-occurrence fallback"""
    try:
        doc = nlp(chunk)
    except Exception as e:
        print(f"spaCy processing error: {e}")
        return {"entities": [], "relationships": []}

    token_ent_map = {}
    entities = []
    seen_entities = set()
    sentence_entities = {}  # Track entities by sentence

    # First pass: extract all entities
    for sent in doc.sents:
        sent_ents = []
        for ent in sent.ents:
            name = ent.text.strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen_entities:
                entities.append({"name": name, "type": ent.label_})
                seen_entities.add(key)
            sent_ents.append(ent)
            for token in ent:
                token_ent_map[token.i] = ent
        sentence_entities[sent] = sent_ents

    relationships = []
    rel_seen = set()  # (source, target, type) tuples

    # Second pass: extract relationships
    for sent, sent_ents in sentence_entities.items():
        # 1. First try verb-based extraction
        for token in sent:
            if token.pos_ not in {"VERB", "AUX"}:
                continue

            subjects = []
            objects = []
            attr_token = None

            for child in token.children:
                ent = token_ent_map.get(child.i)
                if child.dep_ in ("nsubj", "nsubjpass") and ent:
                    subjects.append(ent)
                elif child.dep_ in ("dobj", "dative", "attr", "oprd") and ent:
                    objects.append(ent)
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        pobj_ent = token_ent_map.get(pobj.i)
                        if pobj_ent:
                            objects.append(pobj_ent)
                elif child.dep_ == "conj" and ent:
                    objects.append(ent)

                if child.dep_ == "attr":
                    attr_token = child

            subjects = _dedupe_spans(subjects)
            objects = _dedupe_spans(objects)

            if attr_token and not objects:
                for attr_child in attr_token.children:
                    if attr_child.dep_ == "prep":
                        for pobj in attr_child.children:
                            pobj_ent = token_ent_map.get(pobj.i)
                            if pobj_ent:
                                objects.append(pobj_ent)
                objects = _dedupe_spans(objects)

            if not subjects or not objects:
                continue

            rel_type = token.lemma_.lower()
            if rel_type == "be" and attr_token:
                rel_type = attr_token.text.strip()
            if not rel_type:
                rel_type = "related_to"

            for subj_ent in subjects:
                for obj_ent in objects:
                    if subj_ent == obj_ent:
                        continue
                    source = subj_ent.text.strip()
                    target = obj_ent.text.strip()
                    if not source or not target:
                        continue
                    key = (source.lower(), target.lower(), rel_type.lower())
                    if key in rel_seen:
                        continue
                    rel_seen.add(key)
                    relationships.append({
                        "source": source,
                        "target": target,
                        "type": rel_type
                    })

        # 2. Add co-occurrence relationships for all entity pairs in the sentence
        if len(sent_ents) > 1:
            for i, ent1 in enumerate(sent_ents):
                for ent2 in sent_ents[i+1:]:
                    source = ent1.text.strip()
                    target = ent2.text.strip()
                    if not source or not target or source == target:
                        continue
                    # Ensure consistent ordering for deduplication
                    if source.lower() > target.lower():
                        source, target = target, source
                    key = (source.lower(), target.lower(), "co_occurs_with")
                    if key not in rel_seen:
                        rel_seen.add(key)
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": "co_occurs_with"
                        })

    return {"entities": entities, "relationships": relationships}

# ---------------------------------------------
# STEP 3 & 4: Build Graph & Merge
# ---------------------------------------------
def build_graph_in_neo4j(chunk_items):
    """Build knowledge graph in Neo4j"""
    def _build(session: Session):
        for item in chunk_items:
            for ent in item["entities"]:
                if ent.get("name") and ent.get("type"):
                    session.run(
                        """
                        MERGE (n:Entity {name:$name})
                        SET n.id = COALESCE(n.id, $name),
                            n.type = $type
                        """,
                        name=ent["name"].strip(),
                        type=ent["type"].strip()
                    )

            for rel in item["relationships"]:
                if rel.get("source") and rel.get("target") and rel.get("type"):
                    session.run(
                        """
                        MATCH (s:Entity {name:$source})
                        MATCH (t:Entity {name:$target})
                        MERGE (s)-[r:RELATION]->(t)
                        SET r.type = $rtype
                        """,
                        source=rel["source"].strip(),
                        target=rel["target"].strip(),
                        rtype=rel["type"].strip()
                    )

    _with_session(_build)

# ---------------------------------------------
# STEP 5: Community Detection (NetworkX Louvain)
# ---------------------------------------------
def detect_communities():
    """Detect communities using NetworkX Louvain clustering."""
    print("Running NetworkX Louvain community detection...")

    def _fetch_graph(session: Session):
        node_query = """
        MATCH (n:Entity)
        RETURN elementId(n) AS element_id, n.name AS name, n.type AS type
        """
        edge_query = """
        MATCH (a:Entity)-[:RELATION]->(b:Entity)
        RETURN elementId(a) AS source, elementId(b) AS target
        """

        node_records = session.run(node_query).data()
        edge_records = session.run(edge_query).data()
        return node_records, edge_records

    node_records, edge_records = _with_session(_fetch_graph)

    if not node_records:
        return {"communities": [], "metrics": {"louvain": {"communityCount": 0}}}

    graph = nx.Graph()
    for record in node_records:
        node_id = record["element_id"]
        graph.add_node(
            node_id,
            name=record.get("name"),
            type=record.get("type"),
        )

    for record in edge_records:
        source = record.get("source")
        target = record.get("target")
        if source is None or target is None:
            continue
        graph.add_edge(source, target)

    if graph.number_of_nodes() == 0:
        return {"communities": [], "metrics": {"louvain": {"communityCount": 0}}}

    if graph.number_of_edges() == 0:
        communities_sets = [{node} for node in graph.nodes]
        pagerank_scores = {node: 1.0 for node in graph.nodes}
        modularity_value = 0.0
    else:
        communities_sets = list(nx.algorithms.community.louvain_communities(graph, seed=42))
        pagerank_scores = nx.pagerank(graph)
        modularity_value = nx.algorithms.community.quality.modularity(graph, communities_sets)

    wcc_components = list(nx.connected_components(graph))
    component_map = {
        node: idx for idx, component in enumerate(wcc_components) for node in component
    }

    update_rows: list[dict[str, object]] = []
    for idx, community_nodes in enumerate(communities_sets):
        for node in community_nodes:
            update_rows.append(
                {
                    "element_id": node,
                    "communityId": idx,
                    "pagerank": float(pagerank_scores.get(node, 0.0)),
                    "wccId": component_map.get(node),
                }
            )

    if update_rows:
        def _apply_updates(session: Session):
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:Entity) WHERE elementId(n) = row.element_id
                SET n.{COMMUNITY_PROPERTY} = row.communityId,
                    n.{PAGERANK_PROPERTY} = row.pagerank,
                    n.{WCC_PROPERTY} = row.wccId
                """,
                rows=update_rows,
            )

        _with_session(_apply_updates)

    communities: list[dict[str, object]] = []
    for idx, community_nodes in enumerate(communities_sets):
        ranking = []
        members = []
        for node in community_nodes:
            attrs = graph.nodes[node]
            name = attrs.get("name") or f"entity_{node}"
            members.append(name)
            ranking.append(
                {
                    "name": name,
                    "pagerank": float(pagerank_scores.get(node, 0.0)),
                }
            )

        ranking.sort(key=lambda entry: (-entry["pagerank"], entry["name"]))
        top_nodes = ranking[:COMMUNITY_SEED_LIMIT]
        wcc_ids = sorted({component_map.get(node) for node in community_nodes if node in component_map})

        communities.append(
            {
                "id": f"louvain_{idx}",
                "size": len(community_nodes),
                "members": members[:SUMMARY_TOP_N],
                "top_nodes": top_nodes,
                "wcc_ids": wcc_ids,
            }
        )

    metrics = {
        "wcc": {"componentCount": len(wcc_components)},
        "louvain": {
            "communityCount": len(communities_sets),
            "modularity": modularity_value,
        },
        "pagerank": {
            "nodeCount": graph.number_of_nodes(),
            "edgeCount": graph.number_of_edges(),
        },
    }

    return {"communities": communities, "metrics": metrics}

# ---------------------------------------------
# STEP 6: Community Summarization
# ---------------------------------------------
def summarize_community(community):
    """Summarize a community using its top-ranked nodes."""
    members = community.get("members", [])
    top_nodes = community.get("top_nodes", [])

    if not members:
        return "Empty community"

    names = [node["name"] if isinstance(node, dict) else node for node in members]
    text = ", ".join(names[:SUMMARY_TOP_N])

    highlights = ", ".join(
        f"{node['name']} (PR={node.get('pagerank', 0.0):.4f})"
        for node in top_nodes
    ) if top_nodes else ""

    prompt = f"""Summarize this community of related entities in 2-3 sentences:
Entities: {text}

Key Entities by PageRank: {highlights}

Provide a concise summary describing what these entities represent and their relationships."""
    
    summary = llm_call(prompt)
    return summary if summary else f"Community with entities: {text[:100]}"

# ---------------------------------------------
# STEP 7: Embedding Communities
# ---------------------------------------------

def embed_community(community):
    """Generate and store embedding for a community."""
    if not community or "summary" not in community:
        return
    
    try:
        # Generate embedding for the community summary
        embedding = get_embedding(community["summary"])
        community["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        return True
    except Exception as e:
        print(f"Error embedding community: {e}")
        return False
def embed_communities(communities):
    """Embed community summaries into vector database"""
    for community in communities:
        # This will handle both the embedding and storing in the community dict
        if not embed_community(community):
            continue
            
        doc_id = str(community.get("id", f"comm_{id(community)}"))
        
        try:
            # Store in ChromaDB if available, otherwise use the existing vector_db
            if 'chroma_client' in globals() and chroma_client is not None:
                chroma_collection.upsert(
                    ids=[doc_id],
                    documents=[community.get("summary", "")],
                    metadatas=[{"type": "community", "id": doc_id}],
                    embeddings=[community["embedding"]]
                )
            else:
                # Fallback to the original vector_db implementation
                try:
                    vector_db.delete(ids=[doc_id])
                except Exception:
                    pass
                vector_db.add(
                    ids=[doc_id], 
                    documents=[community.get("summary", "")], 
                    embeddings=[community["embedding"]]
                )
        except Exception as e:
            print(f"Embedding error for community {doc_id}: {e}")

# ---------------------------------------------
# STEP 8: Save Index to Disk
# ---------------------------------------------
def save_index(communities, metrics):
    """Save the built index to disk"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    index_data = {
        "communities": communities,
        "metrics": metrics
    }
    
    with open(INDEX_FILE, "w") as f:
        json.dump(index_data, f, indent=2)
    
    print(f"💾 Index saved to {INDEX_FILE}")

def load_index():
    """Load the pre-built index from disk"""
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            f"Index not found at {INDEX_FILE}. Please run build_index() first."
        )
    
    with open(INDEX_FILE, "r") as f:
        index_data = json.load(f)
    
    print(f"✅ Index loaded from {INDEX_FILE}")
    raw_communities = index_data.get("communities", [])
    normalized = []

    for idx, community in enumerate(raw_communities):
        if isinstance(community, dict):
            normalized.append(community)
            continue

        if isinstance(community, list):
            normalized.append({
                "id": f"legacy_{idx}",
                "members": community,
                "top_nodes": [],
                "summary": ", ".join(community[:SUMMARY_TOP_N]) if community else "",
            })
            continue

        # Fallback for unexpected formats
        normalized.append({
            "id": f"legacy_{idx}",
            "members": [str(community)],
            "top_nodes": [],
            "summary": str(community),
        })

    return normalized, index_data.get("metrics", {})


def index_exists() -> bool:
    return os.path.exists(INDEX_FILE)

# ---------------------------------------------
# STEP 9: Vector Retrieval
# ---------------------------------------------
def retrieve_topk(query, top_k=5):
    """Retrieve top-k most relevant communities"""
    try:
        query_emb = embedder.encode(query).tolist()
        results = vector_db.query(query_embeddings=[query_emb], n_results=top_k)
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        return [
            {"id": results["ids"][0][i],
             "summary": results["documents"][0][i],
             "score": results["distances"][0][i]}
            for i in range(len(results["ids"][0]))
        ]
    except Exception as e:
        print(f"Vector retrieval error: {e}")
        return []

# ---------------------------------------------
# STEP 10: Graph Reasoning (Multi-hop)
# ---------------------------------------------
def graph_reasoning_multi_hop(seed_communities, hops=2, neighbor_limit=25):
    """Expand graph context using community-constrained multi-hop traversal."""
    if not seed_communities:
        return []

    community_prop = COMMUNITY_PROPERTY
    pagerank_prop = PAGERANK_PROPERTY

    nodes_map: dict[str, dict] = {}

    def _update_node(name: str, pagerank: float | None, community_id: str):
        if not name:
            return
        key = (community_id, name)
        current = nodes_map.get(key)
        value = pagerank or 0.0
        if current is None or value > current.get("pagerank", 0.0):
            nodes_map[key] = {
                "name": name,
                "pagerank": value,
                "communityId": community_id,
            }

    def _traverse(session: Session):
        for community in seed_communities:
            if not community:
                continue
            community_id = str(community.get("id"))

            seeds = [
                node.get("name")
                for node in community.get("top_nodes", [])
                if isinstance(node, dict) and node.get("name")
            ]

            if not seeds:
                seeds = [name for name in community.get("members", [])][:COMMUNITY_SEED_LIMIT]

            for seed_name in seeds:
                if not seed_name:
                    continue

                # add the seed itself with known PageRank if available
                pr_lookup = next(
                    (node.get("pagerank") for node in community.get("top_nodes", [])
                     if isinstance(node, dict) and node.get("name") == seed_name),
                    None,
                )
                _update_node(seed_name, pr_lookup, community_id)

                result = session.run(
                    f"""
                    MATCH (seed:Entity {{name:$seed}})
                    MATCH (seed)-[:RELATION*1..{hops}]-(neighbor:Entity)
                    WHERE neighbor.{community_prop} = seed.{community_prop}
                    RETURN DISTINCT neighbor.name AS name,
                           neighbor.{pagerank_prop} AS pagerank
                    ORDER BY pagerank DESC, name ASC
                    LIMIT $limit
                    """,
                    seed=seed_name,
                    limit=neighbor_limit,
                )

                for record in result:
                    _update_node(
                        record.get("name"),
                        record.get("pagerank"),
                        community_id,
                    )

    _with_session(_traverse)

    nodes = list(nodes_map.values())
    nodes.sort(key=lambda item: (-item.get("pagerank", 0.0), item.get("name", "")))
    return nodes

# ---------------------------------------------
# STEP 11: Context Fusion
# ---------------------------------------------
def fuse_context(retrieved, graph_nodes):
    """Fuse retrieved summaries and graph nodes into context"""
    fused = ["Retrieved Community Summaries:"]
    fused += [f"- {r['summary']}" for r in retrieved]
    fused.append("\nCommunity-Neighborhood Entities:")

    for node in graph_nodes[:30]:
        name = node.get("name") if isinstance(node, dict) else str(node)
        pr = node.get("pagerank") if isinstance(node, dict) else None
        community_id = node.get("communityId") if isinstance(node, dict) else None
        fused.append(
            f"- {name}"
            + (f" (PR={pr:.4f})" if pr is not None else "")
            + (f" [community {community_id}]" if community_id is not None else "")
        )

    return "\n".join(fused)

# ---------------------------------------------
# STEP 12: LLM Answer Generation
# ---------------------------------------------
def generate_answer(query, fused_context):
    """Generate final answer using LLM"""
    template = _load_prompt_template("graph_rag_prompt.md")
    prompt = (
        f"{template}\n\n"
        f"Context:\n{fused_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    return llm_call(prompt)

# ---------------------------------------------
# VISUALIZATION
# ---------------------------------------------
def visualize_knowledge_graph(output_html="graph.html", limit=500):
    """Generate an interactive HTML visualization of the knowledge graph."""
    if Network is None:
        raise RuntimeError(
            "pyvis is required for visualization. Install it with 'pip install pyvis'."
        )

    net = Network(height="700px", width="100%", directed=True, notebook=False)
    net.toggle_physics(True)

    node_seen = set()

    with driver.session(database=NEO4J_DATABASE) as session:
        rel_result = session.run(
            """
            MATCH (s:Entity)-[r:RELATION]->(t:Entity)
            RETURN s.name AS source, s.type AS source_type,
                   t.name AS target, t.type AS target_type,
                   r.type AS rel_type
            LIMIT $limit
            """,
            limit=limit,
        )

        for record in rel_result:
            source = record["source"]
            target = record["target"]
            rel_type = record["rel_type"]
            source_group = record["source_type"] or "Entity"
            target_group = record["target_type"] or "Entity"

            if source not in node_seen:
                net.add_node(source, label=source, group=source_group)
                node_seen.add(source)
            if target not in node_seen:
                net.add_node(target, label=target, group=target_group)
                node_seen.add(target)

            net.add_edge(source, target, label=rel_type)

        isolated = session.run(
            """
            MATCH (n:Entity)
            WHERE NOT (n)--()
            RETURN n.name AS name, n.type AS type
            LIMIT $limit
            """,
            limit=limit,
        )

        for record in isolated:
            name = record["name"]
            if name in node_seen:
                continue
            net.add_node(name, label=name, group=record["type"] or "Entity")

    net.show(output_html)
    return output_html

# ---------------------------------------------
# PHASE 1: BUILD INDEX (RUN ONCE)
# ---------------------------------------------
def verify_neo4j_connection() -> dict:
    """Verify Neo4j connection and return node/relationship counts."""
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

def build_index(text, clear_db=True):
    """Build the GraphRAG index with verification and batch processing"""
    print("="*60)
    print("🔨 BUILDING GRAPHRAG INDEX")
    print("="*60)
    
    # Verify initial connection
    print("\nVerifying Neo4j connection...")
    initial_counts = verify_neo4j_connection()
    
    if clear_db:
        print("\nClearing existing graph...")
        clear_graph()
    
    # STEP 1: Semantic Chunking
    print("\nStep 1: Chunking text...")
    chunks = semantic_chunk(text)
    print(f"Created {len(chunks)} chunks")
    
    # STEP 2: Entity & Relationship Extraction
    print("\nStep 2: Extracting entities and relationships...")
    chunk_items = [extract_entities_relations(c) for c in chunks]
    total_entities = sum(len(item["entities"]) for item in chunk_items)
    total_relations = sum(len(item["relationships"]) for item in chunk_items)
    print(f"Extracted {total_entities} entities and {total_relations} relationships")
    
    # STEP 3 & 4: Build Graph
    print("\nStep 3-4: Building graph in Neo4j...")
    build_graph_in_neo4j(chunk_items)
    
    # STEP 5: Community Detection
    print("\nStep 5: Detecting communities...")
    community_payload = detect_communities()
    communities = community_payload.get("communities", [])
    metrics = community_payload.get("metrics", {})
    print(f"Detected {len(communities)} communities")
    
    # STEP 6: Process communities in batches
    print("\nStep 6: Processing communities in batches...")
    batch_size = 5
    delay_between_batches = 3  # seconds
    
    for i in range(0, len(communities), batch_size):
        batch = communities[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(communities) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (communities {i+1}-{min(i+batch_size, len(communities))})")
        
        for community in batch:
            # Attach supporting texts
            attach_supporting_texts([community], chunks)
            
            # Summarize community
            community["summary"] = summarize_community(community)
            
            # Embed community
            embed_community(community)
        
        # Add delay between batches if not the last batch
        if i + batch_size < len(communities):
            print(f"Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
    
    # STEP 7: Save Index
    print("\nStep 7: Saving index to disk...")
    save_index(communities, metrics)
    
    # Verify final state
    print("\nFinal graph statistics:")
    final_counts = verify_neo4j_connection()
    
    # Calculate added nodes/relationships
    if "error" not in initial_counts and "error" not in final_counts:
        added_nodes = final_counts["nodes"] - initial_counts.get("nodes", 0)
        added_rels = final_counts["relationships"] - initial_counts.get("relationships", 0)
        print(f"  • Added {added_nodes} nodes and {added_rels} relationships")
    
    # Print WCC and PageRank metrics if available
    wcc = metrics.get("wcc", {})
    if wcc:
        print(f"  • WCC components: {wcc.get('componentCount', 'n/a')}")
    pr = metrics.get("pagerank", {})
    if pr:
        print(f"  • PageRank iterations: {pr.get('ranIterations', 'n/a')}")
    
    print("\n" + "="*60)
    print("✅ INDEX BUILT SUCCESSFULLY!")
    print("="*60)
    
    return communities, metrics
# PHASE 2: QUERY (RUN MANY TIMES - FAST!)
# ---------------------------------------------
def query_graphrag(query_text, top_k=5, hops=3):
    """Query the pre-built GraphRAG index with increased context"""
    print("="*60)
    print(f"🔍 QUERYING: {query_text}")
    print("="*60)
    
    # Load index if needed (only happens once)
    try:
        communities, metrics = load_index()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please build the index first by running 'rebuild graph'")
        return "Index not found. Please build the index first."

    community_map = {}
    for idx, community in enumerate(communities):
        cid = community.get("id") if isinstance(community, dict) else None
        if cid is None:
            cid = f"legacy_{idx}"
            if isinstance(community, dict):
                community["id"] = cid
            else:
                community = {"id": cid, "members": community}
        community_map[str(cid)] = community

    # Encode query
    query_embedding = get_embedding(query_text)
    
    # Get relevant communities
    retrieved = retrieve_topk(query_text, top_k)
    print(f"Retrieved {len(retrieved)} communities")
    
    # Get relevant entities with increased neighbor limit
    seed_ids = [str(item.get("id")) if hasattr(item, "get") else str(item) for item in retrieved]
    seed_communities = [community_map.get(cid) for cid in seed_ids]
    seed_communities = [c for c in seed_communities if c]
    graph_nodes = graph_reasoning_multi_hop(seed_communities, hops)
    print(f"Collected {len(graph_nodes)} related entities")

    # STEP 11: Context Fusion
    print("\nStep 11: Fusing context...")
    fused_context = fuse_context(retrieved, graph_nodes)
    
    # STEP 12: Answer Generation
    print("\nStep 12: Generating answer...")
    answer = generate_answer(query_text, fused_context)
    
    print("\n" + "="*60)
    return answer


def graph_rag_pipeline(text, query, clear_db=True):
    """Backward-compatible wrapper to build and query the GraphRAG pipeline."""
    if clear_db or not index_exists():
        build_index(text, clear_db=clear_db)
    else:
        print("Using existing GraphRAG index (skipping rebuild).")
    answer = query_graphrag(query)
    return answer or ""

# ---------------------------------------------
# Example Usage
# ---------------------------------------------
if __name__ == "__main__":
    sample_text = """
    Elon Musk is the CEO of Tesla. He also founded SpaceX.
    Tesla produces electric cars and is leading the EV revolution.
    SpaceX launches rockets and aims to colonize Mars.
    Tesla's headquarters is in Austin, Texas.
    SpaceX has successfully launched the Falcon 9 rocket multiple times.
    Elon Musk is also involved with Neuralink and The Boring Company.
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-index', action='store_true', 
                        help='Build the GraphRAG index (run once)')
    parser.add_argument('--query', type=str, 
                        help='Query to ask')
    args = parser.parse_args()
    
    # PHASE 1: Build Index (run once)
    if args.build_index:
        build_index(sample_text)
    
    # PHASE 2: Query (run many times - fast!)
    elif args.query:
        answer = query_graphrag(args.query)
        if answer:
            print("\n💬 ANSWER:")
            print("-" * 60)
            print(answer)
            print("="*60)
    
    # Interactive mode
    else:
        # Check if index exists
        if not os.path.exists(f"{INDEX_DIR}/index.json"):
            print("No index found. Building index for the first time...\n")
            build_index(sample_text)
        
        print("\n🤖 Interactive Query Mode")
        print("Type 'exit' to quit")
        print("-"*60)
        
        while True:
            query = input("\n❓ Question: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            answer = query_graphrag(query)
            if answer:
                print(f"\n💬 Answer: {answer}")
    
    # Close driver
    _close_neo4j_driver()