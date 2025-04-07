import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from environments import Environment2d, OpenSpace2d, RandomSamplePassage, CarParkingEnv, PlanarMobileArm
from space import PointRobot, PolygonalRobot, PlanarMobileArm
from matplotlib.collections import LineCollection
import time
from heapq import heappop, heappush
from state import NumpyState
from utils import smooth_path
from path import Path
import threading 

class PRM():
    def __init__(self, env, num_samples=10, num_neighbors=10, validate_edges=False):
        self.env = env
        self.validate_edges = validate_edges

        self.num_samples = num_samples
        self.num_neighbors = num_neighbors

    def generate_sample_points(self, starting_samples=[]):
        points = np.array([self.env.sample_valid_point().value for _ in range(self.num_samples)] + 
                          [sample for sample in starting_samples if self.env.is_valid(sample)])
        return points

    def create_graph(self, starting_samples=[]):
        vertices = self.generate_sample_points(starting_samples=starting_samples)
        self.graph = Graph(vertices=vertices, num_neighbors=self.num_neighbors)
        # if self.validate_edges:
            # self.validate_graph_edges()

    def validate_graph_edges(self):
        self.invalid_edges = []
        for a, neighbors in enumerate(self.graph.edges):
            for i, b in enumerate(neighbors):
                if not self.env.is_valid_edge(self.graph.vertices[a], self.graph.vertices[b]):
                    self.graph.edges[a, i] = -1
                    self.invalid_edges.append((self.graph.vertices[a], self.graph.vertices[b]))

    def draw(self, ax, path=None, plot_invalid_edges=False, show_task=False):
        self.graph.draw(ax)
        if plot_invalid_edges:
            ax.add_collection(LineCollection(self.invalid_edges, color='orange'))
        if show_task:
            ax.scatter(self.start.value[0], self.start.value[1], s=100, color='green')
            ax.scatter(self.target.value[0], self.target.value[1], s=100, color='red')
        if path:
            path = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
            ax.add_collection(LineCollection(path, color='red'))
    
    def validate_node_edges(self, vertex_idx):
        for neighbors in enumerate(self.graph.edges[vertex_idx]):
            for i, b in enumerate(neighbors):
                if not self.env.is_valid_edge(self.graph.vertices[vertex_idx], self.graph.vertices[b]):
                    self.graph.edges[vertex_idx, i] = -1
                    self.invalid_edges.append((self.graph.vertices[vertex_idx], self.graph.vertices[b]))
        
    def search(self, start : NumpyState, target : NumpyState):
        self.start = start 
        self.target = target

        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

        if self.validate_edges:
            self.validate_graph_edges()

        # Solve with A* or Dijkstra's Algorithm
        path = self.graph.dijkstra_search(start=start.value, end=target.value)
        if path:
            path = Path([self.env.make_state(p) for p in path])
        # Return final Path
        return path

class LazyPRM(PRM):
    def __init__(self, env, num_samples=10, num_neighbors=10):
        super().__init__(env, num_samples=num_samples, num_neighbors=num_neighbors, validate_edges=False)
    
    def lazy_search(self, start : NumpyState, end : NumpyState):
        q = []
        start_idx = self.graph.vertex_to_idx[tuple(start.value)]
        end_idx = self.graph.vertex_to_idx[tuple(end.value)]
        visited = {}
        
        heappush(q, (0, start_idx, None))
        while q:
            dist, node, parent = heappop(q)
            
            if node == end_idx:
                visited[node] = parent
                return self.graph.backtrack(visited, node)

            if node in visited:
                continue
            
            visited[node] = parent

            for i, nidx in enumerate(self.graph.edges[node]):
                if nidx != -1:
                    edge_is_valid = self.env.is_valid_edge(self.graph.vertices[nidx], self.graph.vertices[node])
                    if edge_is_valid:
                        ext_dist = np.linalg.norm(self.graph.vertices[nidx] - self.graph.vertices[node])
                        heappush(q, (dist + ext_dist, nidx, node))
                    else:
                        self.graph.edges[node, i] = -1
        return None
        
    def search(self, start, target):
        self.start = start 
        self.target = target

        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

        # Solve with A* or Dijkstra's Algorithm
        path = self.lazy_search(start=start, end=target)
        if path:
            path = [self.env.make_state(p) for p in path]
        # Return final Path
        return path
    
if __name__ == "__main__":
    # seed = np.random.randint(0, 100)
    seed = 15
    # seed = 13 # Connected Graph Fails to Find Path
    # seed = 66 # Connected Graph Fails to Find Path
    print(f"Seed: {seed}")
    np.random.seed(seed)
    # env = Environment2d()
    # env = RandomSamplePassage()
    # env = PlanarMobileArm()
    # env = OpenSpace2d()
    # env = PointRobot()
    # env = PolygonalRobot()
    env = PlanarMobileArm()
    start, target = env.sample_task()
    # env = CarParkingEnv()
    start_time = time.time()
    prm = PRM(env=env, num_samples=1000, num_neighbors=10, validate_edges=True)
    # prm = ParallelPRM(env=env, num_samples=1000, num_neighbors=10, validate_edges=True)
    # prm = LazyPRM(env=env, num_samples=1000)
    prm.create_graph()
    
    # plt.clf()
    # env.draw_environment(plt.gca())
    # prm.draw(plt.gca())
    # plt.show()

    # start = (0,0)
    # target = (9,9)
    # start = env.make_state(np.array([0,0]))
    # target = env.make_state(np.array([9,9]))
    
    # start = env.make_state(np.array([2.0, 2.75, 0]))
    # target = env.make_state(np.array([-3.0, -2.25, 0]))
    
    path = prm.search(start, target)

    end_time = time.time()
    print(f"Search Time: {end_time - start_time}", f"Num Collision Checks: {env.num_collision_checks}")
    plt.clf()
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

    env.animate_path(path, frame_delay=0.5)

    smoothed_path = smooth_path(env, path)
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=smoothed_path, show_task=True, plot_invalid_edges=False)
    plt.show()

