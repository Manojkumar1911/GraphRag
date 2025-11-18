#!/usr/bin/env python3
"""
Visualize Neo4j Graph using NetworkX and Matplotlib
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_graph_data():
    """Fetch graph data from Neo4j"""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") 
    password = os.getenv("NEO4J_PASSWORD")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Query nodes and edges
    query = """
    MATCH (n:Entity)-[r]-(m:Entity)
    RETURN n.name as source, m.name as target, type(r) as relationship,
           n.communityId as source_community, m.communityId as target_community,
           n.pagerank as source_pagerank, m.pagerank as target_pagerank
    LIMIT 200
    """
    
    with driver.session() as session:
        result = session.run(query)
        edges = []
        nodes = {}
        
        for record in result:
            source = record["source"]
            target = record["target"]
            
            # Store node data
            if source not in nodes:
                nodes[source] = {
                    'community': record.get("source_community", 0),
                    'pagerank': record.get("source_pagerank", 0)
                }
            if target not in nodes:
                nodes[target] = {
                    'community': record.get("target_community", 0), 
                    'pagerank': record.get("target_pagerank", 0)
                }
            
            edges.append((source, target, record["relationship"]))
    
    driver.close()
    return nodes, edges

def visualize_graph(nodes, edges):
    """Create visualization using NetworkX"""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_name, attrs in nodes.items():
        G.add_node(node_name, community=attrs['community'], pagerank=attrs['pagerank'])
    
    # Add edges
    for source, target, relationship in edges:
        G.add_edge(source, target, relationship=relationship)
    
    # Create layout
    plt.figure(figsize=(15, 10))
    
    # Use spring layout with pagerank influence
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Color nodes by community
    communities = [attrs['community'] for _, attrs in G.nodes(data=True)]
    node_colors = plt.cm.Set3([c % 12 for c in communities])
    
    # Size nodes by pagerank
    pageranks = [attrs['pagerank'] * 5000 + 100 for _, attrs in G.nodes(data=True)]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=pageranks, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Neo4j Knowledge Graph Visualization\n(Color = Community, Size = PageRank)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"Graph Statistics:")
    print(f"- Total Nodes: {G.number_of_nodes()}")
    print(f"- Total Edges: {G.number_of_edges()}")
    print(f"- Communities: {len(set(communities))}")
    print(f"- Average PageRank: {sum(attrs['pagerank'] for _, attrs in G.nodes(data=True)) / len(nodes):.6f}")

if __name__ == "__main__":
    print("Fetching graph data from Neo4j...")
    nodes, edges = fetch_graph_data()
    
    print(f"Found {len(nodes)} nodes and {len(edges)} edges")
    print("Creating visualization...")
    
    visualize_graph(nodes, edges)
    print("Visualization saved as 'graph_visualization.png'")
