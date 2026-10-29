from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from pqdict import pqdict
from sklearn.neighbors import KDTree

from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState, Path, Tree


class FastMarchingTreeStar:
    def __init__(self, env: RobotSpace, query_radius: float = 0.5):
        self.env: RobotSpace = env
        self.query_radius = query_radius

        self.tree = Tree()
        self.neighbor_cache: dict[NumpyState, set[NumpyState]] = {}

    def generate_batch_points(self, num_points: int = 10000):
        batch_points = [self.env.sample_valid_point() for _ in range(num_points)]
        return batch_points

    def precompute_neighbors(self, batch_points: list[NumpyState]):
        self.batch_points_numpy = np.array([node.value for node in batch_points])

        N, d = self.batch_points_numpy.shape

        kdt = KDTree(self.batch_points_numpy)
        ind, _ = kdt.query_radius(
            self.batch_points_numpy,
            r=self.query_radius,
            return_distance=True,
            sort_results=True,
        )

        for i in range(N):
            idxes = ind[i]

            # Remove the closest point (always first point) which is itself
            idxes = idxes[1:]

            self.neighbor_cache[batch_points[i]] = set(
                [batch_points[idx] for idx in idxes]
            )

    def search(
        self,
        start: NumpyState,
        target: NumpyState,
        max_steps: int = 10000,
        batch_size: int = 4000,
    ):
        self.start = start
        self.target = target

        batch_points = self.generate_batch_points(batch_size)
        batch_points_with_task = batch_points + [self.start, self.target]
        self.precompute_neighbors(batch_points_with_task)

        V_open = pqdict({self.start: 0.0})
        V_unvisited = set(batch_points_with_task)

        z = self.start
        V_unvisited.remove(self.start)
        self.tree.init_tree(z)

        for i in range(max_steps):
            Nz = self.neighbor_cache[z]

            Hnew = set()
            Xnear = V_unvisited.intersection(Nz)

            for x in Xnear:
                Nx = self.neighbor_cache[x]
                Ynear = list(Nx.intersection(V_open.keys()))

                if len(Ynear) > 0:
                    costs = [V_open[y] + self.env.dist(x, y) for y in Ynear]
                    min_idx = np.argmin(costs)
                    y_min = Ynear[min_idx]

                    if self.env.is_valid_edge(y_min, x):
                        self.tree.add_node(y_min, x)
                        Hnew.add((x, costs[min_idx]))
                        V_unvisited.remove(x)

            # Add candidates to the open set
            for node, c2a in Hnew:
                V_open[node] = c2a

            if len(V_open) > 0:
                z, _ = V_open.popitem()
            else:
                print("FMT*: No More Nodes Open to Expansion")
                break

        backtracked_path = self.tree.backtrack(self.target)
        ordered_path = backtracked_path[::-1]
        return Path(path=ordered_path)

    def draw_tree(self, ax, path: Path | None = None, show_task: bool = True):
        self.env.draw_environment(ax)
        self.tree.draw_tree(ax)
        ax.scatter(self.batch_points_numpy[:, 0], self.batch_points_numpy[:, 1], zorder=0)

        if show_task:
            ax.scatter(self.start.value[0], self.start.value[1], s=100, c="green")
            ax.scatter(self.target.value[0], self.target.value[1], s=100, c="red")

        if path:
            path = [
                (path[i].value[:2], path[i + 1].value[:2]) for i in range(len(path) - 1)
            ]
            ax.add_collection(LineCollection(path, color="red"))

if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    # Create Robot & Environment
    env = PointRobot()
    env.set_obstacles(BiasedPassage())

    # Define Task
    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([15.0, 5.0]))

    # Search via FMT* and Plot Tree
    fmt = FastMarchingTreeStar(env)
    path = fmt.search(start, target)
    fmt.draw_tree(plt.gca(), path=path, show_task=True)
    plt.show()
