from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from heapq import heappush, heappop

class Graph():
    def __init__(self, vertices, num_neighbors=5, max_edge_dist=5):
        self.vertices = vertices
        self.num_neighbors=num_neighbors
        self.max_edge_dist = max_edge_dist
        self.connect_edges()

        self.vertex_to_idx = {tuple(v):i for i, v in enumerate(self.vertices)}

    def connect_edges(self):
        self.kdt = KDTree(self.vertices)
        _, ind = self.kdt.query(self.vertices, k=self.num_neighbors+1)
        self.edges = ind[:, 1:]
    
    def draw(self, ax):
        ax.scatter(self.vertices[:, 0], self.vertices[:, 1])
        line = [(self.vertices[a, :2], self.vertices[b, :2]) for a, neighbors in enumerate(self.edges) for b in neighbors if b > 0]
        lines = np.array(line)
        ax.add_collection(LineCollection(lines))

    def add_vertex(self, vertex):
        self.vertex_to_idx[tuple(vertex)] = len(self.vertices)
        self.vertices = np.vstack((self.vertices, vertex))
        self.connect_edges()

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

