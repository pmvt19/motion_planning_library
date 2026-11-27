import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

from motion_planning.search import RRT
from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState


class BiasedSamplingRRT(RRT):
    def __init__(
        self,
        env: RobotSpace,
        biased_points: np.ndarray,
        points_bias: float = 0.4,
        delta: float = 0.5,
    ):
        super().__init__(env=env, delta=delta)
        self.biased_points: np.ndarray = biased_points
        self.points_bias: float = points_bias
        assert len(self.biased_points) > 0, (
            "Must have a greater than 0 number of biased points"
        )

    def select_node(self, goal_bias: float = 0.0) -> tuple[NumpyState, NumpyState]:
        sampling_type = np.random.choice(
            a=["goal_biased", "points_biased", "uniform_random"],
            p=[goal_bias, self.points_bias, (1 - self.points_bias - goal_bias)],
        )
        d = self.target.value.shape[0]

        if sampling_type == "goal_biased":
            sampled_point = self.target
        elif sampling_type == "points_biased":
            sampled_idx = np.random.choice(len(self.biased_points))
            sampled_point = self.biased_points[sampled_idx] + np.random.normal(
                scale=0.5, size=(d,)
            )  # Unclear if the noise should be included
            sampled_point = self.env.make_state(sampled_point)
        elif sampling_type == "uniform_random":
            sampled_point = self.env.sample_valid_point()

        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point


if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=4))
    start, target = env.sample_task()

    biased_points = np.linspace(start=start.value, stop=target.value, num=15)

    rrt = BiasedSamplingRRT(env, biased_points=biased_points)
    path = rrt.search(
        start, target, max_steps=10, goal_bias=0.1, animate_search_tree=False
    )

    env.draw_environment(plt.gca())
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()
