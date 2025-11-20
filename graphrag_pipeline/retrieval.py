"""
Retrieval and Reasoning Operations
- Vector-based community retrieval
- Multi-hop graph reasoning
- Context fusion
"""

from neo4j import Session

from .config import (
    embedder, COMMUNITY_PROPERTY, PAGERANK_PROPERTY, COMMUNITY_SEED_LIMIT
)
from .database import _with_session, vector_db


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def retrieve_topk(query, top_k=5):
    """Retrieve top-k most relevant communities using vector similarity"""
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


# ============================================================
# GRAPH REASONING (MULTI-HOP)
# ============================================================

def graph_reasoning_multi_hop(seed_communities, hops=2, neighbor_limit=25):
    """Expand graph context using community-constrained multi-hop traversal"""
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
            
            # Get seed nodes from community
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
                
                # Add seed node
                pr_lookup = next(
                    (node.get("pagerank") for node in community.get("top_nodes", [])
                     if isinstance(node, dict) and node.get("name") == seed_name),
                    None,
                )
                _update_node(seed_name, pr_lookup, community_id)
                
                # Multi-hop traversal within same community
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


# ============================================================
# CONTEXT FUSION
# ============================================================

def fuse_context(retrieved, graph_nodes):
    """Fuse retrieved summaries and graph nodes into unified context"""
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
