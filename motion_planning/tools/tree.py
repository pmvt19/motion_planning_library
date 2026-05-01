from collections import defaultdict

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
    