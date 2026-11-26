import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from motion_planning.space import RobotSpace
from motion_planning.tools import Graph, NumpyState, Path


class PRM:
    def __init__(
        self,
        env: RobotSpace,
        num_samples: int = 10,
        num_neighbors: int | None = None,
        edge_dist_radius: float | None = None,
        validate_edges: bool = False,
    ):
        self.env: RobotSpace = env
        self.validate_edges: bool = validate_edges

        self.num_samples: int = num_samples
        self.num_neighbors: int | None = num_neighbors
        self.edge_dist_radius: float | None = edge_dist_radius

        # TODO: Verify Types of edge validity cache
        self.edge_validity_cache: dict[tuple[int, int], bool] = {}

        self.cache_graph_edge_validities: bool = False

    def batch_generate_sample_points(self, starting_samples: list = []) -> np.ndarray:
        starting_samples = np.array(starting_samples)
        sampled_points = self.env.batch_sample_points(self.num_samples)
        points = np.concatenate(
            (sampled_points, starting_samples.reshape(-1, self.env.state_dims)), axis=0
        )
        validities = self.env.batch_is_valid(points)
        return points[validities]

    def create_graph(self, starting_samples: list = []) -> None:
        vertices = self.batch_generate_sample_points(starting_samples=starting_samples)
        self.graph = Graph(
            vertices=vertices,
            num_neighbors=self.num_neighbors,
            edge_dist_radius=self.edge_dist_radius,
        )
        if self.validate_edges:
            self.batch_validate_graph_edges()

    def validate_graph_edges(self) -> None:
        self.invalid_edges = []
        for a, neighbors in enumerate(self.graph.edges):
            for i, b in enumerate(neighbors):
                if not self.env.is_valid_edge(
                    self.env.make_state(self.graph.vertices[a]),
                    self.env.make_state(self.graph.vertices[b]),
                ):
                    # self.graph.edges[a, i] = -1
                    self.graph.edges[a][i] = -1
                    self.invalid_edges.append(
                        (self.graph.vertices[a], self.graph.vertices[b])
                    )

    def batch_validate_graph_edges(self) -> None:
        if self.cache_graph_edge_validities:
            return self.batch_validate_graph_edges_cached()
        else:
            return self.batch_validate_graph_edges_uncached()

    def batch_validate_graph_edges_uncached(self) -> None:
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

        invalid_edge_mask = not edge_validities
        invalid_ids = idx_tracker[invalid_edge_mask]

        for parent, child in invalid_ids:
            self.graph.edges[parent].remove(child)
            self.invalid_edges.append((parent, child))

    def batch_validate_graph_edges_cached(self) -> None:
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
                if (a, b) not in self.edge_validity_cache:
                    start_states.append(self.graph.vertices[a])
                    end_states.append(self.graph.vertices[b])
                    idx_tracker.append((a, b))
                else:
                    if not self.edge_validity_cache[(a, b)]:
                        # self.graph.edges[a].remove(b)
                        to_remove.append(b)
            for b in to_remove:
                self.graph.edges[a].remove(b)

        start_states = np.array(start_states)
        end_states = np.array(end_states)
        idx_tracker = np.array(idx_tracker)

        edge_validities = self.env.batch_is_valid_edge(start_states, end_states)

        invalid_edge_mask = not edge_validities
        invalid_ids = idx_tracker[invalid_edge_mask]

        for parent, child in invalid_ids:
            self.graph.edges[parent].remove(child)
            self.edge_validity_cache[(parent, child)] = False

        valid_edge_mask = edge_validities
        valid_ids = idx_tracker[valid_edge_mask]

        for parent, child in valid_ids:
            self.edge_validity_cache[(parent, child)] = True

    def draw(
        self,
        ax: plt.Axes,
        path: Path | None = None,
        plot_invalid_edges: bool = False,
        show_task: bool = False,
    ) -> None:
        self.graph.draw(ax)
        if plot_invalid_edges:
            invalid_edges = [
                (self.graph.vertices[edge[0]], self.graph.vertices[edge[1]])
                for edge in self.invalid_edges
            ]
            ax.add_collection(LineCollection(invalid_edges, color="#a61107"))
        if show_task:
            ax.scatter(self.start.value[0], self.start.value[1], s=100, color="green")
            ax.scatter(self.target.value[0], self.target.value[1], s=100, color="red")
        if path:
            path = [
                (path[i].value[:2], path[i + 1].value[:2]) for i in range(len(path) - 1)
            ]
            ax.add_collection(LineCollection(path, color="red"))

    def search(
        self,
        start: NumpyState,
        target: NumpyState,
        remove_task: bool = False,
    ) -> Path:
        self.start = start
        self.target = target

        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

        # if self.validate_edges:
        #     self.batch_validate_graph_edges()

        # Solve with A* or Dijkstra's Algorithm
        path = self.graph.dijkstra_search(start=start.value, end=target.value)

        if path:
            path = Path([self.env.make_state(p) for p in path])

        if remove_task:
            self.graph.vertex_to_idx()
        # Return final Path
        return path


if __name__ == "__main__":
    from motion_planning.obstacle_sets import ParkingSpace
    from motion_planning.space import ApproximationSpace, PointRobot
    from motion_planning.utils import set_numpy_seed

    set_numpy_seed()

    env = PointRobot()
    env.set_obstacles(ParkingSpace())

    start, target = env.sample_task()

    # PRM with Standard Space
    prm = PRM(env=env, num_samples=5000, num_neighbors=10, validate_edges=True)

    st = time.time()
    prm.create_graph()
    et = time.time()

    print(f"Time to Create PRM Graph with Standard Environment: {et - st} seconds")
    path = prm.search(start, target)
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

    # PRM with ApproximationSpace
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=False)

    start_time = time.time()
    prm = PRM(env=env, num_samples=5000, num_neighbors=10, validate_edges=True)

    st = time.time()
    prm.create_graph()
    et = time.time()

    print(f"Time to Create PRM Graph with Approx Environment: {et - st} seconds")
    path = prm.search(start, target)
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()
