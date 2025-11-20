"""
Core Graph Operations
- Semantic chunking
- Entity and relationship extraction (spaCy)
- Graph building in Neo4j
- Community detection (NetworkX Louvain)
"""

import re
import time
import networkx as nx
from neo4j import Session

from .config import (
    nlp, COMMUNITY_PROPERTY, WCC_PROPERTY, PAGERANK_PROPERTY,
    COMMUNITY_SEED_LIMIT, SUMMARY_TOP_N, SUPPORTING_TEXT_LIMIT
)
from .database import _with_session
from .llm import llm_call


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def semantic_chunk(text, max_len=300):
    """Split text into semantic chunks based on sentences"""
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


# ============================================================
# ENTITY & RELATIONSHIP EXTRACTION
# ============================================================

def _dedupe_spans(spans):
    """Remove duplicate spaCy spans"""
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
    sentence_entities = {}
    
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
    rel_seen = set()
    
    # Second pass: extract relationships
    for sent, sent_ents in sentence_entities.items():
        # 1. Verb-based extraction
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
        
        # 2. Co-occurrence relationships
        if len(sent_ents) > 1:
            for i, ent1 in enumerate(sent_ents):
                for ent2 in sent_ents[i+1:]:
                    source = ent1.text.strip()
                    target = ent2.text.strip()
                    if not source or not target or source == target:
                        continue
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


# ============================================================
# GRAPH BUILDING IN NEO4J
# ============================================================

def build_graph_in_neo4j(chunk_items):
    """Build knowledge graph in Neo4j from extracted entities and relationships"""
    def _build(session: Session):
        # Create entities
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
        
        # Create relationships
        for item in chunk_items:
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


# ============================================================
# COMMUNITY DETECTION (NETWORKX LOUVAIN)
# ============================================================

def detect_communities():
    """Detect communities using NetworkX Louvain clustering"""
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
    
    # Build NetworkX graph
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
    
    # Run community detection and graph algorithms
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
    
    # Update Neo4j with community assignments
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
    
    # Build community structures
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


# ============================================================
# COMMUNITY SUMMARIZATION
# ============================================================

def summarize_community(community):
    """Summarize a community using its top-ranked nodes"""
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


def attach_supporting_texts(communities, chunks):
    """Populate each community with supporting text snippets from the KB"""
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
