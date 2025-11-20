"""
Utility Functions
- Graph visualization (PyVis)
"""

try:
    from pyvis.network import Network
except ImportError:
    Network = None

from .database import _get_neo4j_driver
from .config import NEO4J_DATABASE


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_knowledge_graph(output_html="graph.html", limit=500):
    """Generate an interactive HTML visualization of the knowledge graph"""
    if Network is None:
        raise RuntimeError(
            "pyvis is required for visualization. Install it with 'pip install pyvis'."
        )
    
    net = Network(height="700px", width="100%", directed=True, notebook=False)
    net.toggle_physics(True)
    
    node_seen = set()
    driver = _get_neo4j_driver()
    
    with driver.session(database=NEO4J_DATABASE) as session:
        # Add nodes with relationships
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
        
        # Add isolated nodes
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
