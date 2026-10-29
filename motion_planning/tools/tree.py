from collections import defaultdict

import numpy as np
from matplotlib.collections import LineCollection


class Tree:
    def __init__(self):
        self.tree = defaultdict(list)
        self.child_to_parent = {}

    def get_nodes(self):
        return self.tree.keys()
    
    def init_tree(self, root_node):
        self.root = root_node
        self.tree[root_node] = []
        self.child_to_parent[root_node] = None
    
    def add_node(self, parent_node, child_node):
        self.tree[parent_node].append(child_node)
        self.tree[child_node]
        self.child_to_parent[child_node] = parent_node

    def backtrack(self, end=None):
        if end is None or end not in self.child_to_parent:
            return []

        path = []
        node = end
        while node:
            path.append(node)
            node = self.child_to_parent[node]
        return path

    def draw_tree(self, ax):
        nodes = np.array([node.value for node in self.tree.keys()])
        ax.scatter(nodes[:, 0], nodes[:, 1])

        edges = [
            [(p.value[0], p.value[1]), (c.value[0], c.value[1])]
            for p in self.tree
            for c in self.tree[p]
        ]
        edges = LineCollection(edges, color="blue")
        ax.add_collection(edges)
    