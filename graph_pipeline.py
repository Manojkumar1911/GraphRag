import os
import re
import json
import pickle
import time
from threading import Lock
from typing import Callable, TypeVar
from neo4j import GraphDatabase, Session, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import spacy

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
GEMINI_MIN_INTERVAL_SECONDS = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "4.5"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
GEMINI_BASE_BACKOFF_SECONDS = float(os.getenv("GEMINI_BASE_BACKOFF_SECONDS", "5"))
INDEX_DIR = os.getenv("INDEX_DIR", "./graphrag_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.json")

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError as exc:
    raise RuntimeError(
        f"spaCy model '{SPACY_MODEL}' is not available. Install it via 'python -m spacy download {SPACY_MODEL}'."
    ) from exc

# ---------------------------------------------
# INITIALIZE Gemini-2.0-Flash
# ---------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

_llm_call_lock = Lock()
_last_llm_call_ts = 0.0


def _respect_rate_limit() -> None:
    global _last_llm_call_ts
    with _llm_call_lock:
        elapsed = time.time() - _last_llm_call_ts
        wait_seconds = GEMINI_MIN_INTERVAL_SECONDS - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)


def _register_call_timestamp() -> None:
    global _last_llm_call_ts
    with _llm_call_lock:
        _last_llm_call_ts = time.time()


def _extract_retry_after_seconds(message: str) -> float | None:
    match = re.search(r"retry in ([0-9.]+)s", message)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def llm_call(prompt: str) -> str:
    """Call Gemini LLM and return text"""
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        _respect_rate_limit()
        try:
            response = gemini_model.generate_content(prompt)
            _register_call_timestamp()
            return response.text or ""
        except Exception as e:
            err_msg = str(e)
            print(f"LLM call error (attempt {attempt}/{GEMINI_MAX_RETRIES}): {err_msg}")

            retry_after = _extract_retry_after_seconds(err_msg)
            if retry_after is None:
                retry_after = GEMINI_BASE_BACKOFF_SECONDS * attempt

            if attempt == GEMINI_MAX_RETRIES:
                return ""

            time.sleep(retry_after)

    return ""

# ---------------------------------------------
# INITIALIZE Embeddings
# ---------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        key = (span.start, span.end)
        if key not in seen:
            unique.append(span)
            seen.add(key)
    return unique


def extract_entities_relations(chunk):
    """Extract entities and relationships using spaCy"""
    try:
        doc = nlp(chunk)
    except Exception as e:
        print(f"spaCy processing error: {e}")
        return {"entities": [], "relationships": []}

    token_ent_map = {}
    entities = []
    seen_entities = set()

    for ent in doc.ents:
        name = ent.text.strip()
        if not name:
            continue
        key = name.lower()
        if key not in seen_entities:
            entities.append({"name": name, "type": ent.label_})
            seen_entities.add(key)
        for token in ent:
            token_ent_map[token.i] = ent

    relationships = []
    rel_seen = set()

    for sent in doc.sents:
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
                        "MERGE (n:Entity {name:$name}) SET n.type = $type",
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
# STEP 5: Community Detection (Louvain Algorithm)
# ---------------------------------------------
def detect_communities():
    """Detect communities using simple connected components"""
    def _detect(session: Session):
        try:
            result = session.run("""
                MATCH (n:Entity)
                OPTIONAL MATCH path = (n)-[*]-(m:Entity)
                WITH n, COLLECT(DISTINCT m) AS connected
                RETURN n.name AS name,
                       CASE WHEN SIZE(connected) > 0
                            THEN connected[0].name
                            ELSE n.name
                       END AS community
            """)

            communities = {}
            for record in result:
                comm_id = record["community"]
                communities.setdefault(comm_id, []).append(record["name"])

            return list(communities.values())
        except Exception as e:
            print(f"Community detection error: {e}")
            result = session.run("MATCH (n:Entity) RETURN n.name AS name")
            return [[record["name"] for record in result]]

    return _with_session(_detect)

# ---------------------------------------------
# STEP 6: Community Summarization
# ---------------------------------------------
def summarize_community(nodes):
    """Summarize a community of entities"""
    if not nodes:
        return "Empty community"
    
    text = ", ".join(nodes[:50])
    prompt = f"""Summarize this community of related entities in 2-3 sentences:
Entities: {text}

Provide a concise summary describing what these entities represent and their relationships."""
    
    summary = llm_call(prompt)
    return summary if summary else f"Community with entities: {text[:100]}"

# ---------------------------------------------
# STEP 7: Embedding Communities
# ---------------------------------------------
def embed_communities(summaries):
    """Embed community summaries into vector database"""
    for idx, summary in enumerate(summaries):
        if summary:
            try:
                emb = embedder.encode(summary).tolist()
                vector_db.add(ids=[f"comm_{idx}"], documents=[summary], embeddings=[emb])
            except Exception as e:
                print(f"Embedding error for community {idx}: {e}")

# ---------------------------------------------
# STEP 8: Save Index to Disk
# ---------------------------------------------
def save_index(communities, summaries):
    """Save the built index to disk"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    index_data = {
        "communities": communities,
        "summaries": summaries
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
    return index_data["communities"], index_data["summaries"]


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
def graph_reasoning_multi_hop(retrieved_summaries, hops=2):
    """Expand retrieved nodes using multi-hop graph traversal"""
    nodes_set = set()

    def _traverse(session: Session):
        for summary in retrieved_summaries:
            result = session.run(
                """
                MATCH (n:Entity)
                RETURN n.name AS name
                LIMIT 10
                """
            )

            for record in result:
                node_name = record["name"]
                hop_result = session.run(
                    f"""
                    MATCH path = (n:Entity {{name:$name}})-[*1..{hops}]-(m:Entity)
                    RETURN DISTINCT m.name AS neighbor
                    LIMIT 20
                    """,
                    name=node_name,
                )

                nodes_set.add(node_name)
                nodes_set.update([r["neighbor"] for r in hop_result])

    _with_session(_traverse)

    return list(nodes_set)

# ---------------------------------------------
# STEP 11: Context Fusion
# ---------------------------------------------
def fuse_context(retrieved, graph_nodes):
    """Fuse retrieved summaries and graph nodes into context"""
    fused = ["Retrieved Community Summaries:"]
    fused += [f"- {r['summary']}" for r in retrieved]
    fused.append("\nRelated Entities from Knowledge Graph:")
    fused += [f"- {n}" for n in graph_nodes[:30]]
    return "\n".join(fused)

# ---------------------------------------------
# STEP 12: LLM Answer Generation
# ---------------------------------------------
def generate_answer(query, fused_context):
    """Generate final answer using LLM"""
    prompt = f"""You are an expert at reasoning over a knowledge graph.
Use ONLY the provided context to answer the question.

Context:
{fused_context}

Question: {query}

Provide a clear, concise answer based only on the context above. If the context doesn't contain enough information, say so.

Answer:"""
    
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
def build_index(text, clear_db=True):
    """Build the GraphRAG index - run this ONCE"""
    print("="*60)
    print("🔨 BUILDING GRAPHRAG INDEX (Run once or when data changes)")
    print("="*60)
    
    if clear_db:
        print("Clearing existing graph...")
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
    communities = detect_communities()
    print(f"Detected {len(communities)} communities")
    
    # STEP 6: Community Summarization
    print("\nStep 6: Summarizing communities...")
    summaries = [summarize_community(c) for c in communities]
    
    # STEP 7: Embed Communities
    print("\nStep 7: Embedding communities...")
    embed_communities(summaries)
    
    # STEP 8: Save Index
    print("\nStep 8: Saving index to disk...")
    save_index(communities, summaries)
    
    print("\n" + "="*60)
    print("✅ INDEX BUILT SUCCESSFULLY!")
    print("="*60)
    
    return communities, summaries

# ---------------------------------------------
# PHASE 2: QUERY (RUN MANY TIMES - FAST!)
# ---------------------------------------------
def query_graphrag(query_text, top_k=3, hops=2):
    """Query the pre-built GraphRAG index - FAST!"""
    print("="*60)
    print(f"🔍 QUERYING: {query_text}")
    print("="*60)
    
    # Load index if needed (only happens once)
    try:
        communities, summaries = load_index()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    
    # STEP 9: Vector Retrieval
    print("\nStep 9: Retrieving relevant communities...")
    retrieved = retrieve_topk(query_text, top_k)
    print(f"Retrieved {len(retrieved)} communities")
    
    # STEP 10: Graph Reasoning
    print("\nStep 10: Multi-hop graph reasoning...")
    graph_nodes = graph_reasoning_multi_hop([r["summary"] for r in retrieved], hops)
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