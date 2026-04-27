import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from matplotlib.collections import LineCollection
from heapq import heappush, heappop
from collections import defaultdict

# from motion_planning.utils import issue_warning

class Graph():
    def __init__(self, vertices, num_neighbors=None, edge_dist_radius=None):
        self.vertices = vertices
        self.num_neighbors = num_neighbors
        self.edge_dist_radius = edge_dist_radius

        # issue_warning(self.num_neighbors is not None and self.edge_dist_radius is not None, 'Specified both num_neighbors and edge_dist_radius, defaulting to num_neighbors', 'warning')

        if self.num_neighbors is not None:
            self.connection_strategy = 'knn'
        elif self.edge_dist_radius is not None:
            self.connection_strategy = 'r_neighborhood'
        else:
            raise ValueError("Must Specify num_neighbors or edge_dist_radius")
        print(f"Connecting Edges Using {self.connection_strategy} strategy")
        self.connect_edges()
        self.vertex_to_idx = {tuple(v):i for i, v in enumerate(self.vertices)}

    def get_node_connections(self, vertices):
        if self.connection_strategy == 'knn':
            ind = self.kdt.query(vertices, k=self.num_neighbors+1, return_distance=False)
        elif self.connection_strategy == 'r_neighborhood':
            ind = self.kdt.query_radius(vertices, r=self.edge_dist_radius)
        return ind

    def connect_edges(self):
        self.kdt = KDTree(self.vertices)
        ind = self.get_node_connections(self.vertices)

        self.edges = defaultdict(set)
        for a in range(len(ind)):
            for b in ind[a]:
                if a != b:
                    self.edges[a].add(b)
                    self.edges[b].add(a)
    
    def draw(self, ax):
        ax.scatter(self.vertices[:, 0], self.vertices[:, 1], color='#FFA500')
        line = [(self.vertices[a, :2], self.vertices[b, :2]) for a in self.edges for b in self.edges[a] if b >= 0]
        ax.add_collection(LineCollection(line, color="#cccccc", alpha=0.4))

    def add_vertex(self, vertex):
        self.vertex_to_idx[tuple(vertex)] = len(self.vertices)
        self.vertices = np.vstack((self.vertices, vertex))

        ind = self.get_node_connections(np.array([vertex]))[0]
    
        a = self.vertex_to_idx[tuple(vertex)]

        for b in ind:
            if a != b:
                self.edges[a].add(b)
                self.edges[b].add(a)

    def backtrack(self, visited, end):
        path_idxs = []
        node = end
        while node:
            path_idxs.append(node)
            node = visited[node]

        path = [self.vertices[idx] for idx in path_idxs]
        return path[::-1]
    
    def dijkstra_search(self, start : tuple, end : tuple):
        q = []
        start_idx = self.vertex_to_idx[tuple(start)]
        end_idx = self.vertex_to_idx[tuple(end)]
        visited = {}
        
        heappush(q, (0, start_idx, None))
        while q:
            dist, node, parent = heappop(q)
            
            if node == end_idx:
                visited[node] = parent
                return self.backtrack(visited, node)

            if node in visited:
                continue
            
            visited[node] = parent

            for nidx in self.edges[node]:
                if nidx != -1:
                    ext_dist = np.linalg.norm(self.vertices[nidx] - self.vertices[node])
                    heappush(q, (dist + ext_dist, nidx, node))

        return None

        

if __name__ == '__main__':
    np.random.seed(0)
    vertices = np.random.uniform(-10, 10, size=(1000, 2))
    graph = Graph(vertices)

    graph.connect_edges()

    graph.draw(plt.gca())
    plt.show()

    start = vertices[0]
    target = vertices[-1]

    graph.dijkstra_search(start, target)

