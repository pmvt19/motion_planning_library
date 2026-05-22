import numpy as np

from motion_planning.search import PRM
from motion_planning.tools import Graph, NumpyState, Path


class IncrementalPRM(PRM):
    def __init__(
        self,
        env,
        num_samples=100,
        num_neighbors=None,
        edge_dist_radius=None,
        cache_edge_validities=True,
    ):
        super().__init__(
            env=env,
            num_samples=num_samples,
            num_neighbors=num_neighbors,
            edge_dist_radius=edge_dist_radius,
        )
        self.cache_graph_edge_validities = cache_edge_validities

    def extend_graph(self):
        old_vertices = self.graph.vertices
        new_vertices = self.batch_generate_sample_points()
        vertices = np.vstack((old_vertices, new_vertices))
        self.graph = Graph(
            vertices=vertices,
            num_neighbors=self.num_neighbors,
            edge_dist_radius=self.edge_dist_radius,
        )

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

    def search(self, start: NumpyState, target: NumpyState):
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
    import matplotlib.pyplot as plt

    from motion_planning.obstacle_sets import RandomSamplePassage
    from motion_planning.space import PolygonalRobot
    from motion_planning.utils import set_numpy_seed

    set_numpy_seed()

    env = PolygonalRobot()
    env.set_obstacles(RandomSamplePassage(num_walls=3, gap_width=2))

    prm = IncrementalPRM(env, num_samples=1000, num_neighbors=10)
    prm.create_graph()

    start, target = (
        env.make_state(np.array([5.0, 5.0, 0.0])),
        env.make_state(np.array([35.0, 5.0, 0.0])),
    )

    path = prm.search(start, target)

    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path, show_task=True)
    plt.show()
