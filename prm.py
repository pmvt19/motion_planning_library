import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from space import RobotSpace, PointRobot, PolygonalRobot, PlanarMobileArm
from circle_approximation import ApproximationSpace
from matplotlib.collections import LineCollection
import time
from heapq import heappop, heappush
from state import NumpyState
from utils import smooth_path, interpolate_edge, interpolate_path
from path import Path
# import threading 
from obstacle_sets import TestSet, ParkingSpace, RandomSamplePassage, CentralObstacle, BiasedPassage, WeavingPassage

class PRM():
    def __init__(self, env : RobotSpace, num_samples=10, num_neighbors=None, edge_dist_radius=None, validate_edges=False):
        self.env = env
        self.validate_edges = validate_edges

        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.edge_dist_radius = edge_dist_radius

        self.edge_validity_cache = {}

        self.cache_graph_edge_validities = False

    # def generate_sample_points(self, starting_samples=[]):
    #     points = np.array([self.env.sample_valid_point().value for _ in range(self.num_samples)] + 
    #                       [sample for sample in starting_samples if self.env.is_valid(sample)])
    #     return points

    def batch_generate_sample_points(self, starting_samples=[]):
        points = np.array([self.env.sample_point().value for _ in range(self.num_samples)] + [sample for sample in starting_samples])
        validities = self.env.batch_is_valid(points)
        return points[validities]

    def create_graph(self, starting_samples=[]):
        # vertices = self.generate_sample_points(starting_samples=starting_samples)
        vertices = self.batch_generate_sample_points(starting_samples=starting_samples)
        self.graph = Graph(vertices=vertices, num_neighbors=self.num_neighbors, edge_dist_radius=self.edge_dist_radius)
        # if self.validate_edges:
            # self.batch_validate_graph_edges()

    def validate_graph_edges(self):
        self.invalid_edges = []
        for a, neighbors in enumerate(self.graph.edges):
            for i, b in enumerate(neighbors):
                if not self.env.is_valid_edge(self.env.make_state(self.graph.vertices[a]), self.env.make_state(self.graph.vertices[b])):
                    # self.graph.edges[a, i] = -1
                    self.graph.edges[a][i] = -1
                    self.invalid_edges.append((self.graph.vertices[a], self.graph.vertices[b]))
                
    def batch_validate_graph_edges(self):
        if self.cache_graph_edge_validities:
            return self.batch_validate_graph_edges_cached()
        else:
            return self.batch_validate_graph_edges_uncached()

    def batch_validate_graph_edges_uncached(self):
        """
        use with new, variable number of edges graph representation
        """
        self.invalid_edges = []
        start_states = []
        end_states = []
        idx_tracker = []

        for a in self.graph.edges:
            for b in self.graph.edges[a]:
                start_states.append(self.graph.vertices[a])
                end_states.append(self.graph.vertices[b])
                idx_tracker.append((a, b))
        
        start_states = np.array(start_states)
        end_states = np.array(end_states)
        idx_tracker = np.array(idx_tracker)

        edge_validities = self.env.batch_is_valid_edge(start_states, end_states)

        invalid_edge_mask = (edge_validities == False)
        invalid_ids = idx_tracker[invalid_edge_mask]

        for parent, child in invalid_ids:
            self.graph.edges[parent].remove(child)

    def batch_validate_graph_edges_cached(self):
        """
        use with new, variable number of edges graph representation
        """
        self.invalid_edges = []
        start_states = []
        end_states = []
        idx_tracker = []

        for a in self.graph.edges:
            to_remove = []
            for b in self.graph.edges[a]:
                if (a,b) not in self.edge_validity_cache:
                    start_states.append(self.graph.vertices[a])
                    end_states.append(self.graph.vertices[b])
                    idx_tracker.append((a, b))
                else:
                    if self.edge_validity_cache[(a,b)] == False:
                        # self.graph.edges[a].remove(b)
                        to_remove.append(b)
            for b in to_remove:
                self.graph.edges[a].remove(b)


        
        start_states = np.array(start_states)
        end_states = np.array(end_states)
        idx_tracker = np.array(idx_tracker)

        edge_validities = self.env.batch_is_valid_edge(start_states, end_states)

        invalid_edge_mask = (edge_validities == False)
        invalid_ids = idx_tracker[invalid_edge_mask]

        for parent, child in invalid_ids:
            self.graph.edges[parent].remove(child)
            self.edge_validity_cache[(parent,child)] = False

        valid_edge_mask = (edge_validities == True)
        valid_ids = idx_tracker[valid_edge_mask]

        for parent, child in valid_ids:
            self.edge_validity_cache[(parent,child)] = True

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
        
    def search(self, start : NumpyState, target : NumpyState):
        self.start = start 
        self.target = target

        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

        if self.validate_edges:
            self.batch_validate_graph_edges()

        # Solve with A* or Dijkstra's Algorithm
        path = self.graph.dijkstra_search(start=start.value, end=target.value)

        if path:
            path = Path([self.env.make_state(p) for p in path])
        # Return final Path
        return path

class NonUniformPRM(PRM):
    def __init__(self, env : RobotSpace, num_samples=10, num_neighbors=10, validate_edges=False, scale=1):
        super().__init__(env=env, num_samples=num_samples, num_neighbors=num_neighbors, validate_edges=validate_edges)
        self.scale = scale
    
    def batch_generate_sample_points(self, starting_samples=[]):
        points = np.array([self.env.sample_point().value for _ in range(self.num_samples)] + [sample for sample in starting_samples])
        offset_points = points + np.random.normal(scale=self.scale, size=points.shape)
        points_validities = self.env.batch_is_valid(points)
        offset_points_validities = self.env.batch_is_valid(offset_points)
        xor_validities = np.logical_xor(points_validities, offset_points_validities)
        
        final_validities = np.logical_and(points_validities, xor_validities)
        final_offset_validities = np.logical_and(offset_points_validities, xor_validities)
        return np.vstack((points[final_validities], offset_points[final_offset_validities]))

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
            path = Path([self.env.make_state(p) for p in path])
        # Return final Path
        return path

class IncrementalPRM(PRM):
    def __init__(self, env, num_samples=100, num_neighbors=None, edge_dist_radius=None, cache_edge_validities=True):
        super().__init__(env=env, num_samples=num_samples, num_neighbors=num_neighbors, edge_dist_radius=edge_dist_radius)
        self.cache_graph_edge_validities = cache_edge_validities

    def extend_graph(self):
        old_vertices = self.graph.vertices
        new_vertices = self.batch_generate_sample_points()
        vertices = np.vstack((old_vertices, new_vertices))
        self.graph = Graph(vertices=vertices, num_neighbors=self.num_neighbors, edge_dist_radius=self.edge_dist_radius)

    def _dfs_iterative(self, start, target):
        start_idx = self.graph.vertex_to_idx[tuple(start.value)]
        target_idx = self.graph.vertex_to_idx[tuple(target.value)]
        s = [start_idx]

        visited = set()

        while s:
            node_idx = s.pop()
            if node_idx == target_idx:
                return True
            if node_idx in visited:
                continue
            visited.add(node_idx)

            neighbors = self.graph.edges[node_idx]
            for nbr in neighbors:
                s.append(nbr)
        return False

    def _bfs_iterative(self, start, target):
        start_idx = self.graph.vertex_to_idx[tuple(start.value)]
        target_idx = self.graph.vertex_to_idx[tuple(target.value)]
        q = [start_idx]

        visited = set()

        while q:
            node_idx = q.pop(0)
            if node_idx == target_idx:
                return True
            if node_idx in visited:
                continue
            visited.add(node_idx)

            neighbors = self.graph.edges[node_idx]
            for nbr in neighbors:
                q.append(nbr)
        return False

    def is_nodes_connected(self, start, target):
        # return self._dfs_iterative(start, target)
        return self._bfs_iterative(start, target)

    def search(self, start : NumpyState, target : NumpyState):
        self.start = start 
        self.target = target

        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

        self.batch_validate_graph_edges()
        is_connected = self.is_nodes_connected(start, target)
        num_times_extended = 0
        while not is_connected:
            print(f"Adding Nodes, Current Size: {len(self.graph.vertices)}")
            self.extend_graph()
            self.batch_validate_graph_edges()
            is_connected = self.is_nodes_connected(start, target)
            num_times_extended += 1

        print(f"Num Times Extended: {num_times_extended}")

        # Solve with A* or Dijkstra's Algorithm
        path = self.graph.dijkstra_search(start=start.value, end=target.value)

        if path:
            path = Path([self.env.make_state(p) for p in path])
        # Return final Path
        return path

if __name__ == "__main__":
    seed = np.random.randint(0, 100)
    # seed = 15
    # seed = 37 # Goes through the gap for PolygonRobot and ParkingSpace
    # seed = 41
    # seed = 91
    # seed = 83
    # seed = 46
    seed = 3
    print(f"Seed: {seed}")
    np.random.seed(seed)

    # env = PointRobot()
    env = PolygonalRobot()
    # env = PlanarMobileArm()

    env.set_obstacles(ParkingSpace())
    # env.set_obstacles(WeavingPassage())
    # env.set_obstacles(RandomSamplePassage(gap_width=1.1))
    # env.set_obstacles(BiasedPassage(num_walls=3))
    # env.set_obstacles(CentralObstacle())
    # env.set_obstacles(TestSet())

    start, target = env.sample_task()
    # env = ApproximationSpace(env, batch_size=10000, do_overapproximation=False)
    start_time = time.time()
    # prm = PRM(env=env, num_samples=500, num_neighbors=10, validate_edges=True)
    # prm = PRM(env=env, num_samples=20000, num_neighbors=10, validate_edges=True)
    # prm = PRM(env=env, num_samples=1000, edge_dist_radius=2.4, validate_edges=True)
    # prm = PRM(env=env, num_samples=5000, edge_dist_radius=0.5, validate_edges=True)
    # prm = NonUniformPRM(env=env, num_samples=10000, num_neighbors=10, validate_edges=True)
    # prm = LazyPRM(env=env, num_samples=1000, num_neighbors=10)

    # prm = IncrementalPRM(env=env, num_samples=10000, num_neighbors=5)
    prm = IncrementalPRM(env=env, num_samples=50, edge_dist_radius=5)
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
    print(f"PRM Num Nodes: {len(prm.graph.vertices)}")
    # for p in path: 
    #     print(p.value)

    end_time = time.time()
    print(f"Search Time: {end_time - start_time}", f"Num Collision Checks: {env.num_collision_checks}")

    # # PLOTTING

    plt.clf()
    env.draw_environment(plt.gca())
    # env.space.draw_environment(plt.gca())
    # space.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    # env.space.draw_environment(plt.gca())
    # env.draw_environment(plt.gca())
    plt.show()
    interpolated_path = []
    path = smooth_path(env, path)
    plt.clf()
    env.draw_environment(plt.gca())
    # env.space.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

    # # PLOTTING

    path = interpolate_path(path, env, 0.1)
    # env.space.animate_path(path, frame_delay=0.001)
    # env.animate_path(path, frame_delay=0.001)

    