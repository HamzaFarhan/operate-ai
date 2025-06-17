import matplotlib.pyplot as plt
import networkx as nx
import pytest

from src.operate_ai.memory_tools import Entity, KnowledgeGraph, Relation


def visualize_knowledge_graph(kg: KnowledgeGraph, title: str = "Knowledge Graph", save_path: str | None = None):
    """
    Visualize a KnowledgeGraph using matplotlib and NetworkX.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph to visualize
    title : str
        Title for the plot
    save_path : str | None
        If provided, save the plot to this path instead of showing it
    """
    G = kg.to_networkx()

    if G.number_of_nodes() == 0:
        print("Empty graph - nothing to visualize")
        return

    plt.figure(figsize=(12, 8))

    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Separate entity and observation nodes
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "entity"]
    obs_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "observation"]

    # Draw entity nodes (larger, different color)
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color="lightblue", node_size=1000, alpha=0.8)

    # Draw observation nodes (smaller, different color)
    nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes, node_color="lightgreen", node_size=300, alpha=0.6)

    # Draw edges with different styles for different types
    relation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "relation"]
    obs_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "has_observation"]

    # Draw relation edges (solid, thicker)
    nx.draw_networkx_edges(
        G, pos, edgelist=relation_edges, edge_color="red", width=2, alpha=0.7, arrows=True, arrowsize=20
    )

    # Draw observation edges (dashed, thinner)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=obs_edges,
        edge_color="gray",
        width=1,
        alpha=0.5,
        style="dashed",
        arrows=True,
        arrowsize=15,
    )

    # Add labels for entity nodes only (to avoid clutter)
    entity_labels = {n: n for n in entity_nodes}
    nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=10, font_weight="bold")

    # Add edge labels for relations
    relation_edge_labels = {
        (u, v): d.get("relation_type", "") for u, v, d in G.edges(data=True) if d.get("edge_type") == "relation"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=relation_edge_labels, font_size=8)

    plt.title(title, size=16, weight="bold")
    plt.axis("off")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightblue", label="Entities"),
        Patch(facecolor="lightgreen", label="Observations"),
        plt.Line2D([0], [0], color="red", lw=2, label="Relations"),
        plt.Line2D([0], [0], color="gray", lw=1, linestyle="--", label="Has Observation"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def test_visualization():
    """Demo function showing how to visualize different knowledge graphs."""

    # Simple graph with one entity and observations
    print("=== Simple Graph ===")
    entity1 = Entity(name="Alice", entity_type="person", observations=["CEO", "Tech enthusiast"])
    kg1 = KnowledgeGraph(entities={"Alice": entity1})
    visualize_knowledge_graph(kg1, "Simple Knowledge Graph", "simple_kg.png")

    # Complex graph with multiple entities and relations
    print("\n=== Complex Graph ===")
    alice = Entity(name="Alice", entity_type="person", observations=["CEO", "Founded company"])
    bob = Entity(name="Bob", entity_type="person", observations=["CTO", "Loves Python"])
    company = Entity(name="TechCorp", entity_type="company", observations=["AI startup", "Founded 2020"])
    project = Entity(name="AI Assistant", entity_type="project", observations=["Main product", "Uses LLMs"])

    relations = [
        Relation(relation_from="Alice", relation_to="TechCorp", relation_type="founded"),
        Relation(relation_from="Bob", relation_to="TechCorp", relation_type="works_at"),
        Relation(relation_from="Alice", relation_to="Bob", relation_type="manages"),
        Relation(relation_from="TechCorp", relation_to="AI Assistant", relation_type="develops"),
        Relation(relation_from="Bob", relation_to="AI Assistant", relation_type="leads"),
    ]

    kg2 = KnowledgeGraph(
        entities={"Alice": alice, "Bob": bob, "TechCorp": company, "AI Assistant": project},
        relations={(r.relation_from, r.relation_to, r.relation_type): r for r in relations},
    )

    visualize_knowledge_graph(kg2, "Complex Knowledge Graph", "complex_kg.png")

    print("\nVisualization complete! Check the generated PNG files.")


def test_to_networkx_empty_graph():
    """Test converting an empty knowledge graph to NetworkX."""
    kg = KnowledgeGraph()
    G = kg.to_networkx()

    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_to_networkx_single_entity_no_observations():
    """Test converting a graph with a single entity and no observations."""
    entity = Entity(name="Alice", entity_type="person", observations=[])
    kg = KnowledgeGraph(entities={"Alice": entity})

    G = kg.to_networkx()

    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0

    # Check entity node attributes
    assert G.nodes["Alice"]["node_type"] == "entity"
    assert G.nodes["Alice"]["entity_type"] == "person"
    assert G.nodes["Alice"]["name"] == "Alice"


def test_to_networkx_entity_with_observations():
    """Test converting a graph with an entity that has observations."""
    entity = Entity(name="Alice", entity_type="person", observations=["Works at Tech Corp", "Lives in NYC"])
    kg = KnowledgeGraph(entities={"Alice": entity})

    G = kg.to_networkx()

    # Should have 1 entity + 2 observation nodes = 3 nodes
    assert G.number_of_nodes() == 3
    # Should have 2 edges (entity -> each observation)
    assert G.number_of_edges() == 2

    # Check entity node
    assert G.nodes["Alice"]["node_type"] == "entity"

    # Check observation nodes
    obs_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "observation"]
    assert len(obs_nodes) == 2

    for obs_node in obs_nodes:
        assert G.nodes[obs_node]["parent_entity"] == "Alice"
        assert G.nodes[obs_node]["observation"] in ["Works at Tech Corp", "Lives in NYC"]

    # Check edges from entity to observations
    entity_edges = list(G.out_edges("Alice", data=True))
    assert len(entity_edges) == 2
    for _, _, edge_data in entity_edges:
        assert edge_data["edge_type"] == "has_observation"


def test_to_networkx_multiple_entities_with_relations():
    """Test converting a graph with multiple entities and relations."""
    alice = Entity(name="Alice", entity_type="person", observations=["CEO"])
    bob = Entity(name="Bob", entity_type="person", observations=["Engineer"])
    company = Entity(name="Tech Corp", entity_type="company", observations=["Founded 2020"])

    relation1 = Relation(relation_from="Alice", relation_to="Tech Corp", relation_type="works_at")
    relation2 = Relation(relation_from="Bob", relation_to="Tech Corp", relation_type="works_at")
    relation3 = Relation(relation_from="Alice", relation_to="Bob", relation_type="manages")

    kg = KnowledgeGraph(
        entities={"Alice": alice, "Bob": bob, "Tech Corp": company},
        relations={
            ("Alice", "Tech Corp", "works_at"): relation1,
            ("Bob", "Tech Corp", "works_at"): relation2,
            ("Alice", "Bob", "manages"): relation3,
        },
    )

    G = kg.to_networkx()

    # 3 entities + 3 observations = 6 nodes
    assert G.number_of_nodes() == 6
    # 3 observation edges + 3 relation edges = 6 edges
    assert G.number_of_edges() == 6

    # Check entity nodes
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "entity"]
    assert len(entity_nodes) == 3
    assert set(entity_nodes) == {"Alice", "Bob", "Tech Corp"}

    # Check relation edges
    relation_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("edge_type") == "relation"]
    assert len(relation_edges) == 3

    relation_types = {(u, v): d["relation_type"] for u, v, d in relation_edges}
    assert relation_types[("Alice", "Tech Corp")] == "works_at"
    assert relation_types[("Bob", "Tech Corp")] == "works_at"
    assert relation_types[("Alice", "Bob")] == "manages"


def test_to_networkx_observation_node_naming():
    """Test that observation nodes have unique, predictable names."""
    entity = Entity(name="Test Entity", entity_type="test", observations=["First obs", "Second obs", "Third obs"])
    kg = KnowledgeGraph(entities={"Test Entity": entity})

    G = kg.to_networkx()

    # Check observation node IDs
    obs_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "observation"]
    expected_obs_nodes = ["Test Entity__obs_0", "Test Entity__obs_1", "Test Entity__obs_2"]

    assert set(obs_nodes) == set(expected_obs_nodes)

    # Check that observations are in the right order
    for i, expected_node in enumerate(expected_obs_nodes):
        assert G.nodes[expected_node]["observation"] == entity.observations[i]


def test_to_networkx_missing_networkx():
    """Test that ImportError is raised when NetworkX is not available."""
    from unittest.mock import patch

    # Mock missing networkx by patching the import
    with patch.dict("sys.modules", {"networkx": None}):
        kg = KnowledgeGraph()
        with pytest.raises(ImportError, match="NetworkX is required"):
            kg.to_networkx()


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    test_to_networkx_empty_graph()
    test_to_networkx_single_entity_no_observations()
    test_to_networkx_entity_with_observations()
    test_to_networkx_multiple_entities_with_relations()
    test_to_networkx_observation_node_naming()
    test_visualization()
    print("All tests passed!")
