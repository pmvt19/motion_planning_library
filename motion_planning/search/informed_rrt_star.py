import numpy as np
from sklearn.neighbors import KDTree

from motion_planning.search import RRTStar
from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState, Path


class InformedRRTStar(RRTStar):
    def __init__(
        self,
        env: RobotSpace,
        delta: float = 0.5,
        rewire_radius: float = 1,
        max_rewire_neighbors: int = 20,
    ):
        super().__init__(
            env=env,
            delta=delta,
            rewire_radius=rewire_radius,
            max_rewire_neighbors=max_rewire_neighbors,
        )

    # TODO: Potentially Optimize This Loop
    def sample_point_in_ellipse(self, path_length: float) -> NumpyState:
        sampled_point = self.env.sample_valid_point()

        while (
            self.env.dist(sampled_point, self.start)
            + self.env.dist(sampled_point, self.target)
            > path_length
        ):
            sampled_point = self.env.sample_valid_point()

        return sampled_point

    # TODO: Potentially Optimize This Loop
    def compute_path_length(self, path: Path) -> float:
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.env.dist(path[i], path[i + 1])
        return path_length

    def select_node(self, goal_bias=0):
        path = self.backtrack(end=target)
        if len(path) > 0:
            path_length = self.compute_path_length(path)
            sampled_point = self.sample_point_in_ellipse(path_length)
        elif np.random.random() < goal_bias:
            sampled_point = self.target
        else:
            sampled_point = self.env.sample_valid_point()
        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))
    rrt = InformedRRTStar(env)

    start, target = (
        env.make_state(np.array([5.0, 5.0])),
        env.make_state(np.array([15.0, 5.0])),
    )
    path = rrt.search(start, target, max_steps=5000)

    rrt.draw_tree(plt.gca(), path=path)
    plt.show()
