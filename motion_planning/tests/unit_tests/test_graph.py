import numpy as np
import matplotlib.pyplot as plt

from motion_planning.tools import Graph

def test_graph_num_nodes_generated():
    np.random.seed(0)
    vertices = np.random.uniform(-10, 10, size=(1000, 2))

    graph = Graph(vertices)
    graph.connect_edges()

    assert(len(graph.vertices) == 1000)
    assert(len(graph.edges) > 0)
    assert(graph.connection_strategy == 'knn')

def test_graph_edge_connection_strategy_hierarchy():
    vertices = np.random.uniform(-10, 10, size=(1000, 2))

    graph = Graph(vertices, num_neighbors=10)
    assert(graph.connection_strategy == 'knn')

    graph = Graph(vertices, edge_dist_radius=3)
    assert(graph.connection_strategy == 'r_neighborhood')

    graph = Graph(vertices, num_neighbors=10, edge_dist_radius=3)
    assert(graph.connection_strategy == 'knn')