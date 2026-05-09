import unittest

import numpy as np

from motion_planning.tools import Graph


class TestGraph(unittest.TestCase):

    def test_graph_num_nodes_generated(self):
        np.random.seed(0)
        vertices = np.random.uniform(-10, 10, size=(1000, 2))

        graph = Graph(vertices, num_neighbors=10)
        graph.connect_edges()

        self.assertEqual(len(graph.vertices), 1000)
        self.assertGreater(len(graph.edges), 0)
        self.assertEqual(graph.connection_strategy, 'knn')

    def test_graph_edge_connection_strategy_hierarchy(self):
        vertices = np.random.uniform(-10, 10, size=(1000, 2))

        graph = Graph(vertices, num_neighbors=10)
        self.assertEqual(graph.connection_strategy, 'knn')

        graph = Graph(vertices, edge_dist_radius=3)
        self.assertEqual(graph.connection_strategy, 'r_neighborhood')

        graph = Graph(vertices, num_neighbors=10, edge_dist_radius=3)
        self.assertEqual(graph.connection_strategy, 'knn')