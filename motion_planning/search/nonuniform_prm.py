import numpy as np

from motion_planning.search import PRM
from motion_planning.space import RobotSpace


class NonUniformPRM(PRM):
    def __init__(
        self,
        env: RobotSpace,
        num_samples: int = 10,
        num_neighbors: int = 10,
        validate_edges: bool = False,
        scale: float = 1.0,
    ):
        super().__init__(
            env=env,
            num_samples=num_samples,
            num_neighbors=num_neighbors,
            validate_edges=validate_edges,
        )
        self.scale: float = scale

    def batch_generate_sample_points(
        self, starting_samples: list[np.ndarray] = []
    ) -> np.ndarray:
        starting_samples: np.ndarray = np.array(starting_samples)
        sampled_points = self.env.batch_sample_points(self.num_samples)
        points = np.concatenate(
            (sampled_points, starting_samples.reshape(-1, self.env.state_dims)), axis=0
        )
        offset_points = points + np.random.normal(scale=self.scale, size=points.shape)
        points_validities = self.env.batch_is_valid(points)
        offset_points_validities = self.env.batch_is_valid(offset_points)
        xor_validities = np.logical_xor(points_validities, offset_points_validities)

        final_validities = np.logical_and(points_validities, xor_validities)
        final_offset_validities = np.logical_and(
            offset_points_validities, xor_validities
        )
        return np.vstack(
            (points[final_validities], offset_points[final_offset_validities])
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from motion_planning.obstacle_sets import RandomSamplePassage
    from motion_planning.space import PolygonalRobot
    from motion_planning.utils import set_numpy_seed

    set_numpy_seed(7903)

    env = PolygonalRobot()
    env.set_obstacles(RandomSamplePassage(num_walls=3, gap_width=2))

    prm = NonUniformPRM(env, num_samples=1000, num_neighbors=10, validate_edges=True)
    prm.create_graph()

    start, target = (
        env.make_state(np.array([5.0, 5.0, 0.0])),
        env.make_state(np.array([35.0, 5.0, 0.0])),
    )

    path = prm.search(start, target)

    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path, show_task=True)
    plt.show()
