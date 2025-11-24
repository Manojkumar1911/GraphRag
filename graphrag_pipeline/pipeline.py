"""
Main Pipeline Orchestration
- build_index(): Build GraphRAG index from text
- query_graphrag(): Query the pre-built index
- graph_rag_pipeline(): Backward-compatible wrapper
"""

import os
import json
import time

from .config import (
    INDEX_DIR, INDEX_FILE, SUMMARY_TOP_N, get_embedding
)
from .database import verify_neo4j_connection, clear_graph, vector_db
from .graph_operations import (
    semantic_chunk, extract_entities_relations, build_graph_in_neo4j,
    detect_communities, summarize_community, attach_supporting_texts
)
from .retrieval import retrieve_topk, graph_reasoning_multi_hop, fuse_context
from .llm import generate_answer


# ============================================================
# EMBEDDING COMMUNITIES
# ============================================================

def embed_community(community):
    """Generate and store embedding for a community"""
    if not community or "summary" not in community:
        return False
    
    try:
        embedding = get_embedding(community["summary"])
        community["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        return True
    except Exception as e:
        print(f"Error embedding community: {e}")
        return False


def embed_communities(communities):
    """Embed community summaries into vector database"""
    for community in communities:
        # Generate embedding
        if not embed_community(community):
            continue
            
        doc_id = str(community.get("id", f"comm_{id(community)}"))
        
        try:
            # Delete old entry if exists, then add new one
            try:
                vector_db.delete(ids=[doc_id])
            except Exception:
                pass
            
            # Add to ChromaDB vector store
            vector_db.add(
                ids=[doc_id], 
                documents=[community.get("summary", "")], 
                embeddings=[community["embedding"]]
            )
        except Exception as e:
            print(f"Embedding error for community {doc_id}: {e}")


# ============================================================
# INDEX MANAGEMENT
# ============================================================

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
    """Check if index file exists"""
    return os.path.exists(INDEX_FILE)


# ============================================================
# PHASE 1: BUILD INDEX (RUN ONCE)
# ============================================================

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
    chunk_items = []
    for chunk in chunks:
        extracted = extract_entities_relations(chunk)
        chunk_items.append({"text": chunk, **extracted})
    
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
    
    # STEP 6: Process communities in batches (summarize only)
    print("\nStep 6: Summarizing communities in batches...")
    batch_size = 5
    delay_between_batches = 3  # seconds
    
    for i in range(0, len(communities), batch_size):
        batch = communities[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(communities) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (communities {i+1}-{min(i+batch_size, len(communities))})")
        
        for community in batch:
            # Summarize community
            community["summary"] = summarize_community(community["id"])
            
            # Embed community
            embed_community(community)
        
        # Add delay between batches if not the last batch
        if i + batch_size < len(communities):
            print(f"Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
    
    # STEP 7: Attach supporting texts ONCE (outside loop)
    print("\nStep 7: Attaching supporting texts to entities...")
    attach_supporting_texts(chunk_items)
    
    # STEP 8: Embed all communities into vector DB
    print("\nStep 8: Storing embeddings in vector database...")
    embed_communities(communities)
    
    # STEP 9: Save Index
    print("\nStep 9: Saving index to disk...")
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


# ============================================================
# PHASE 2: QUERY (RUN MANY TIMES - FAST!)
# ============================================================

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
    
    # Build community map
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
    
    # Retrieve relevant communities
    retrieved = retrieve_topk(query_text, top_k)
    print(f"Retrieved {len(retrieved)} communities")
    
    # Get relevant entities via multi-hop reasoning
    seed_ids = [str(item.get("id")) if hasattr(item, "get") else str(item) for item in retrieved]
    seed_communities = [community_map.get(cid) for cid in seed_ids]
    seed_communities = [c for c in seed_communities if c]
    graph_nodes = graph_reasoning_multi_hop(seed_communities, hops)
    print(f"Collected {len(graph_nodes)} related entities")
    
    # Fuse context
    print("\nStep 11: Fusing context...")
    fused_context = fuse_context(retrieved, graph_nodes)
    
    # Generate answer
    print("\nStep 12: Generating answer...")
    answer = generate_answer(query_text, fused_context)
    
    print("\n" + "="*60)
    return answer


# ============================================================
# BACKWARD-COMPATIBLE WRAPPER
# ============================================================

def graph_rag_pipeline(text, query, clear_db=True):
    """Backward-compatible wrapper to build and query the GraphRAG pipeline"""
    if clear_db or not index_exists():
        build_index(text, clear_db=clear_db)
    else:
        print("Using existing GraphRAG index (skipping rebuild).")
    answer = query_graphrag(query)
    return answer or ""