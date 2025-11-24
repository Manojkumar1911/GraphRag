"""
Optimized & Fixed: Graph Operations with DeBERTa v3
- Fixes: duplicated embedding calls, batching, robust imports, safer Neo4j writes,
  improved relation detection, defensive checks and logging.
- Author: assistant (fixed version)
"""

import re
import logging
from collections import defaultdict
from typing import List, Dict, Optional
from itertools import combinations

import torch
import networkx as nx
from neo4j import Session
from transformers import pipeline, AutoTokenizer, AutoModel

# Try safe relative imports; if running as script, fall back to absolute
try:
    from .config import (
        COMMUNITY_PROPERTY, PAGERANK_PROPERTY,
        SUMMARY_TOP_N, SUPPORTING_TEXT_LIMIT
    )
    from .database import _with_session
    from .llm import llm_call
except Exception:
    # fallback: try importing from top-level (useful when running as __main__)
    try:
        from config import (
            COMMUNITY_PROPERTY, PAGERANK_PROPERTY,
            SUMMARY_TOP_N, SUPPORTING_TEXT_LIMIT
        )
        from database import _with_session
        from llm import llm_call
    except Exception as e:
        raise ImportError("Required local modules not found. Ensure config.py, database.py and llm.py exist and are importable.") from e

# ============================================================
# SETUP
# ============================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models lazily
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NER pipeline - try DeBERTa NER then fallback
try:
    ner_pipeline = pipeline("ner", model="Davlan/deberta-v3-base-ner",
                           aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    logger.warning(f"DeBERTa NER failed, using BERT: {e}")
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER",
                           aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

# Tokenizer + model
try:
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    deberta_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device).eval()
    logger.info(f"DeBERTa model loaded on {device}")
except Exception as e:
    logger.error(f"Failed to load DeBERTa model: {e}")
    raise

# ============================================================
# TEXT PROCESSING
# ============================================================

def semantic_chunk(text: str, max_len: int = 300) -> List[str]:
    """Split text into semantic chunks by sentences, respecting approximate character length."""
    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_len:
            current += (" " + sent if current else sent)
        else:
            if current:
                chunks.append(current)
            # If single sentence is longer than max_len, still keep it as single chunk
            if len(sent) > max_len:
                chunks.append(sent)
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks

# ============================================================
# ENTITY & RELATIONSHIP EXTRACTION
# ============================================================

def extract_entities_ner(text: str, threshold: float = 0.7) -> List[Dict]:
    """Extract entities using NER with confidence filtering and deduplication."""
    if not text or not text.strip():
        return []

    try:
        results = ner_pipeline(text)
        entities, seen = [], set()

        for ent in results:
            # pipeline may return 'word' or 'entity'
            name = ent.get('word') or ent.get('entity') or ''
            name = str(name).strip()

            score = float(ent.get('score', 0.0))
            ent_type = ent.get('entity_group') or ent.get('entity') or 'UNKNOWN'

            if (len(name) < 2 or score < threshold or
                not re.search(r'[a-zA-Z0-9]', name) or name.lower() in seen):
                continue

            entities.append({
                'name': name,
                'type': ent_type,
                'score': score
            })
            seen.add(name.lower())

        return entities
    except Exception as e:
        logger.error(f"NER error: {e}")
        return []


@torch.no_grad()
def get_embeddings(texts: List[str], max_length: int = 512) -> Dict[str, Optional[torch.Tensor]]:
    """Batch embeddings for a list of texts. Returns dict text->embedding (cpu tensors)."""
    out = {}
    if not texts:
        return out

    # Deduplicate inputs preserving order
    seen_order = []
    seen_set = set()
    for t in texts:
        if t not in seen_set:
            seen_set.add(t)
            seen_order.append(t)

    try:
        # Tokenize all at once with padding
        inputs = deberta_tokenizer(seen_order, return_tensors="pt", padding=True,
                                   truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = deberta_model(**inputs)
        last_hidden = outputs.last_hidden_state  # (batch, seq, dim)

        # Mean pooling with attention mask
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        embeddings = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

        # Move to CPU and map back
        embeddings = embeddings.cpu()
        for idx, txt in enumerate(seen_order):
            out[txt] = embeddings[idx]

        # Cleanup
        del inputs, outputs, last_hidden, mask, embeddings
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        # fallback to per-text embedding
        for t in seen_order:
            try:
                inp = deberta_tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
                inp = {k: v.to(device) for k, v in inp.items()}
                out_t = deberta_model(**inp).last_hidden_state
                mask_t = inp['attention_mask'].unsqueeze(-1).expand(out_t.size()).float()
                emb = torch.sum(out_t * mask_t, dim=1) / torch.clamp(mask_t.sum(dim=1), min=1e-9)
                out[t] = emb.squeeze(0).cpu()
                del inp, out_t, mask_t, emb
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e2:
                logger.warning(f"Failed to embed single text: {e2}")
                out[t] = None

    return out


def detect_relation_type(sentence: str, ent1: str, ent2: str) -> str:
    """Detect relationship type using pattern matching between entity spans."""
    sent_lower = sentence.lower()
    ent1_l, ent2_l = ent1.lower(), ent2.lower()
    pos1, pos2 = sent_lower.find(ent1_l), sent_lower.find(ent2_l)

    if pos1 < 0 or pos2 < 0:
        return "related_to"

    # Calculate substring strictly between entity spans
    start = min(pos1, pos2) + len(ent1_l if pos1 < pos2 else ent2_l)
    end = max(pos1, pos2)
    between = sent_lower[start:end].strip()

    # Sanitize short separators
    if re.match(r'^[,\s]*(and|or|,)[,\s]*$', between):
        between = ''

    patterns = {
        'works_at': r'\b(works?\s+at|employed\s+by|hired\s+by|works\s+for)\b',
        'located_in': r'\b(located\s+in|based\s+in|headquarter(?:ed)?\s+in)\b',
        'founded': r'\b(founded|established|created|co-founded|cofounder)\b',
        'owns': r'\b(owns?|owned|ownership|owner\b)\b',
        'manages': r'\b(manages?|oversees?|managed)\b',
        'acquired': r'\b(acquired|bought|purchased|acquisition)\b',
        'leads': r'\b(leads?|heads?|headed)\b',
        'partner_with': r'\b(partner(?:s|ed)?\s+with|collaborates?\s+with)\b',
        'ceo_of': r'\b(ceo\s+of|chief\s+executive\s+officer\s+of|is\s+ceo\s+of)\b',
        'member_of': r'\b(member\s+of|part\s+of)\b',
        'reports_to': r'\b(reports?\s+to|reporting\s+to)\b'
    }

    for rel_type, pattern in patterns.items():
        if re.search(pattern, between):
            return rel_type

    # If no clear pattern but short between text, maybe appositive relation
    if len(between) <= 40 and between:
        return 'related_to'

    return "related_to"


def extract_relationships(text: str, entities: List[Dict],
                         sim_threshold: float = 0.65, max_entities: int = 50) -> List[Dict]:
    """Extract relationships via co-occurrence and semantic similarity (fixed and optimized)."""
    if len(entities) < 2:
        return []

    # Limit entities
    if len(entities) > max_entities:
        entities = sorted(entities, key=lambda x: x.get('score', 0), reverse=True)[:max_entities]

    relationships, seen = [], set()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # Strategy 1: Co-occurrence in sentences
    for sentence in sentences:
        sent_lower = sentence.lower()
        sent_entities = [e for e in entities if e['name'].lower() in sent_lower]

        for i, ent1 in enumerate(sent_entities):
            for ent2 in sent_entities[i+1:]:
                key = tuple(sorted([ent1['name'].lower(), ent2['name'].lower()]))

                pos1 = sent_lower.find(ent1['name'].lower())
                pos2 = sent_lower.find(ent2['name'].lower())
                if pos1 < 0 or pos2 < 0:
                    continue
                start = min(pos1, pos2) + len(ent1['name'] if pos1 < pos2 else ent2['name'])
                end = max(pos1, pos2)
                between = sent_lower[start:end].strip()

                # allow longer between but filter extremely long noise
                if re.match(r'^[,\s]*(and|or|,)[,\s]*$', between):
                    continue
                if len(between) > 300:
                    # very long context; skip co-occurrence (likely not a direct relation)
                    continue

                if key not in seen:
                    seen.add(key)
                    relationships.append({
                        'source': ent1['name'],
                        'target': ent2['name'],
                        'type': detect_relation_type(sentence, ent1['name'], ent2['name']),
                        'score': 1.0,
                        'evidence': sentence[:500]
                    })

    # Strategy 2: Semantic similarity (batch embeddings)
    entity_ctx = {}
    for ent in entities:
        for sent in sentences:
            if ent['name'].lower() in sent.lower():
                entity_ctx[ent['name']] = sent
                break

    unique_sents = list({v: None for v in entity_ctx.values()}.keys())
    sent_embs = get_embeddings(unique_sents)

    entity_names = list(entity_ctx.keys())
    for i, ent1 in enumerate(entity_names):
        for ent2 in entity_names[i+1:]:
            key = tuple(sorted([ent1.lower(), ent2.lower()]))
            if key in seen:
                continue

            s1, s2 = entity_ctx.get(ent1), entity_ctx.get(ent2)
            e1, e2 = sent_embs.get(s1), sent_embs.get(s2)
            if e1 is not None and e2 is not None:
                try:
                    sim = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0), dim=1).item()
                except Exception as e:
                    logger.warning(f"Cosine similarity error: {e}")
                    continue

                if sim >= sim_threshold:
                    seen.add(key)
                    relationships.append({
                        'source': ent1,
                        'target': ent2,
                        'type': 'semantically_related',
                        'score': float(sim),
                        'evidence': f"Context similarity: {sim:.4f}"
                    })

    logger.info(f"Extracted {len(relationships)} relationships from {len(entities)} entities")
    return relationships


def extract_entities_relations(chunk: str) -> Dict[str, List]:
    """Main extraction pipeline for a single chunk."""
    try:
        entities = extract_entities_ner(chunk)
        if not entities:
            return {"entities": [], "relationships": []}

        relationships = extract_relationships(chunk, entities)

        # Deduplicate relationships by (source,target,type)
        unique_rels, seen = [], set()
        for rel in relationships:
            key = tuple(sorted([rel['source'].lower(), rel['target'].lower()])) + (rel.get('type', 'related_to'),)
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)

        # Return
        return {"entities": entities, "relationships": unique_rels}
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return {"entities": [], "relationships": []}

# ============================================================
# NEO4J GRAPH OPERATIONS
# ============================================================

def build_graph_in_neo4j(chunk_items: List[Dict]):
    """Build or update knowledge graph in Neo4j with defensive checks."""
    def _build(session: Session):
        try:
            # Create indexes (Neo4j 4.4+ supports IF NOT EXISTS)
            try:
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            except Exception:
                # older server versions may not support IF NOT EXISTS
                try:
                    session.run("CREATE INDEX ON :Entity(name)")
                except Exception:
                    pass

            # Collect unique entities
            entity_batch, seen_ents = [], set()
            for item in chunk_items:
                for ent in item.get("entities", []):
                    name = (ent.get("name") or "").strip()
                    if name and name.lower() not in seen_ents:
                        entity_batch.append({
                            "name": name,
                            "type": ent.get("type", "UNKNOWN"),
                            "score": float(ent.get("score", 0.0))
                        })
                        seen_ents.add(name.lower())

            # Batch insert entities
            if entity_batch:
                session.run("""
                    UNWIND $entities AS ent
                    MERGE (n:Entity {name: ent.name})
                    ON CREATE SET n.type = ent.type, n.id = ent.name,
                                  n.confidence = ent.score, n.created_at = timestamp()
                    ON MATCH SET n.confidence = CASE WHEN ent.score > coalesce(n.confidence,0) 
                                                     THEN ent.score ELSE n.confidence END
                """, entities=entity_batch)
                logger.info(f"Inserted/updated {len(entity_batch)} entities")

            # Collect unique relationships
            rel_batch, seen_rels = [], set()
            for item in chunk_items:
                for rel in item.get("relationships", []):
                    src, tgt = (rel.get("source") or "").strip(), (rel.get("target") or "").strip()
                    rtype = (rel.get("type") or "related_to").strip()

                    if src and tgt and rtype:
                        key = tuple(sorted([src.lower(), tgt.lower()])) + (rtype.lower(),)
                        if key not in seen_rels:
                            rel_batch.append({
                                "source": src, "target": tgt, "rtype": rtype,
                                "score": float(rel.get("score", 1.0)),
                                "evidence": (rel.get("evidence") or "")[:500]
                            })
                            seen_rels.add(key)

            # Batch insert relationships
            if rel_batch:
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (s:Entity {name: rel.source}), (t:Entity {name: rel.target})
                    MERGE (s)-[r:RELATION {type: rel.rtype}]->(t)
                    ON CREATE SET r.confidence = rel.score, r.evidence = rel.evidence,
                                  r.created_at = timestamp()
                    ON MATCH SET r.confidence = CASE WHEN rel.score > coalesce(r.confidence,0)
                                                     THEN rel.score ELSE r.confidence END
                """, rels=rel_batch)
                logger.info(f"Inserted/updated {len(rel_batch)} relationships")

        except Exception as e:
            logger.error(f"Graph build error: {e}")
            raise

    return _with_session(_build)


def detect_communities(resolution: float = 1.0):
    """Detect communities using Louvain algorithm and persist community ids + pagerank."""
    def _detect(session: Session):
        try:
            # Fetch graph
            result = session.run("""
                MATCH (s:Entity)-[r:RELATION]->(t:Entity)
                RETURN s.name as source, t.name as target, coalesce(r.confidence, 1.0) as weight
            """)

            G = nx.Graph()
            for rec in result:
                src = rec.get("source")
                tgt = rec.get("target")
                w = rec.get("weight") or 1.0
                if src and tgt:
                    G.add_edge(src, tgt, weight=float(w))

            if not G.nodes():
                return {"communities": [], "metrics": {"node_count": 0, "edge_count": 0}}

            logger.info(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

            # Louvain community detection (networkx provides louvain_communities)
            communities = nx.community.louvain_communities(G, resolution=resolution, seed=42)

            # Write communities to Neo4j
            for comm_id, community in enumerate(communities):
                session.run(f"""
                    UNWIND $nodes AS node_name
                    MATCH (n:Entity {{name: node_name}})
                    SET n.{COMMUNITY_PROPERTY} = $comm_id
                """, nodes=list(community), comm_id=comm_id)

            # Compute and write PageRank
            pagerank = nx.pagerank(G, weight='weight')
            session.run(f"""
                UNWIND $batch AS item
                MATCH (n:Entity {{name: item.name}})
                SET n.{PAGERANK_PROPERTY} = item.score
            """, batch=[{"name": n, "score": float(s)} for n, s in pagerank.items()])

            logger.info(f"Detected {len(communities)} communities")

            # Build community records
            community_records = []
            for comm_id, community in enumerate(communities):
                top_nodes = list(session.run(f"""
                    MATCH (n:Entity) WHERE n.{COMMUNITY_PROPERTY} = $comm_id
                    RETURN n.name AS name, n.{PAGERANK_PROPERTY} AS pagerank
                    ORDER BY CASE WHEN n.{PAGERANK_PROPERTY} IS NULL THEN 0 ELSE n.{PAGERANK_PROPERTY} END DESC

                    LIMIT $limit
                """, comm_id=comm_id, limit=SUMMARY_TOP_N))

                community_records.append({
                    "id": comm_id,
                    "members": list(community),
                    "top_nodes": [{"name": n["name"], "pagerank": n.get("pagerank")} 
                                  for n in top_nodes if n.get("name")]
                })

            return {
                "communities": community_records,
                "metrics": {
                    "node_count": len(G.nodes()),
                    "edge_count": len(G.edges()),
                    "wcc": {"componentCount": len(list(nx.connected_components(G)))},
                    "pagerank": {"ranIterations": len(pagerank)}
                }
            }
        except Exception as e:
            logger.error(f"Community detection error: {e}")
            raise

    return _with_session(_detect)


def summarize_community(community_id: int) -> str:
    """Generate LLM summary for community."""
    def _summarize(session: Session) -> str:
        try:
            # Get top entities
            entities = list(session.run(f"""
                MATCH (n:Entity) WHERE n.{COMMUNITY_PROPERTY} = $comm_id
                RETURN n.name as name, n.type as type
                ORDER BY n.{PAGERANK_PROPERTY} DESC
                LIMIT $limit
            """, comm_id=community_id, limit=SUMMARY_TOP_N))

            if not entities:
                return f"Community {community_id}: No entities found"

            # Get relationships (sample)
            rels = list(session.run(f"""
                MATCH (s:Entity)-[r:RELATION]->(t:Entity)
                WHERE s.{COMMUNITY_PROPERTY} = $comm_id AND t.{COMMUNITY_PROPERTY} = $comm_id
                RETURN s.name as source, r.type as type, t.name as target
                LIMIT 50
            """, comm_id=community_id))

            prompt = f"""Summarize this knowledge graph community:\n\nKey Entities: {', '.join(f"{e['name']} ({e['type']})" for e in entities)}\n\nKey Relationships:\n{chr(10).join(f"{r['source']} --[{r['type']}]--> {r['target']}" for r in rels)}\n\nProvide a concise 2-3 sentence summary."""

            return llm_call(prompt)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return f"Community {community_id}: Error generating summary"

    return _with_session(_summarize)


def attach_supporting_texts(chunk_items: List[Dict]):
    """Attach supporting text snippets to entity nodes (defensive)."""
    def _attach(session: Session):
        try:
            entity_texts = defaultdict(list)

            for item in chunk_items:
                snippet = (item.get("text", "") or "").strip()[:300]
                if not snippet:
                    continue
                for ent in item.get("entities", []):
                    name = (ent.get("name") or "").strip()
                    if name and snippet not in entity_texts[name] and len(entity_texts[name]) < SUPPORTING_TEXT_LIMIT:
                        entity_texts[name].append(snippet)

            if not entity_texts:
                return

            for name, texts in entity_texts.items():
                session.run("MATCH (n:Entity {name: $name}) SET n.supporting_texts = coalesce(n.supporting_texts, []) + $texts",
                           name=name, texts=texts)

            logger.info(f"Attached supporting texts to {len(entity_texts)} entities")
        except Exception as e:
            logger.error(f"Supporting text error: {e}")
            raise

    return _with_session(_attach)

# End of file
