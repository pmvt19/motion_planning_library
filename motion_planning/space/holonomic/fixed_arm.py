import numpy as np
from shapely import LineString

from motion_planning.controller.xbox_controller import XboxController
from motion_planning.space import HolonomicRobot
from motion_planning.tools import AngularNumpyState
from motion_planning.utils import numpystate_distance


class FixedArm(HolonomicRobot):
    def __init__(
        self,
        egde_validity_delta: float = 0.5,
        arm_link_lengths: np.ndarray = np.array([2, 2]),
    ):
        super().__init__()
        self.angular_dims_start = 0
        self.edge_validity_delta = egde_validity_delta
        self.arm_link_lengths = arm_link_lengths
        self.num_links = len(self.arm_link_lengths)

    def dist(self, state1, state2) -> float:
        return numpystate_distance(state1, state2)

    def sample_point(self) -> AngularNumpyState:
        thetas = np.random.uniform(
            low=np.zeros(self.num_links),
            high=np.ones(self.num_links) * (2 * np.pi),
            size=(self.num_links,)
        )
        return self.make_state(thetas)

    def make_state(self, state) -> AngularNumpyState:
        return AngularNumpyState(
            value=state, angular_dims_start=self.angular_dims_start
        )

    def generate_robot_representation(self, state):
        end_points = self.forward_kinematics(state)
        lines = []
        for i in range(len(end_points) - 1):
            line = LineString([end_points[i], end_points[i + 1]])
            lines.append(line)
        return lines

    def is_self_colliding(self, state):
        lines = self.generate_robot_representation(state)
        for i in range(len(lines)):
            for j in range(i + 2, len(lines)):
                if lines[i].intersects(lines[j]):
                    return True
        return False

    def is_valid(self, state) -> bool:
        self.num_collision_checks += 1
        lines = self.generate_robot_representation(state)

        if self.is_self_colliding(state):
            return False

        for obs in self.obstacles:
            for line in lines:
                if obs.intersects(line):
                    return False

        return True

    def forward_kinematics(self, state) -> np.ndarray:
        thetas = self.get_state_value(state)

        cum_thetas = np.cumsum(thetas)
        cos_vals = np.cos(cum_thetas)
        sin_vals = np.sin(cum_thetas)

        uncoordinated_points = np.stack(
            (cos_vals, sin_vals), axis=1
        ) * self.arm_link_lengths.reshape(-1, 1)
        uncoordinated_points = np.vstack((np.zeros(2), uncoordinated_points))
        end_points = np.cumsum(uncoordinated_points, axis=0)
        return end_points

    def draw_state(self, ax, state):
        lines = self.generate_robot_representation(state)
        for line in lines:
            ax.plot(*line.xy, color="blue")

    def input_to_x_dot(self, inputs) -> np.ndarray:
        dt = 0.1
        assert len(self.arm_link_lengths) == 2, (
            "User Input Only Implemented for Robot with 2 Arms"
        )

        theta1_dot = inputs[XboxController.XboxControls.LTHUMBX] * dt
        theta2_dot = -inputs[XboxController.XboxControls.LTHUMBY] * dt

        return np.array([theta1_dot, theta2_dot])

    ## ---- Batched Methods ---- ##
    def batch_forward_kinematics(self, states) -> np.ndarray:
        B = states.shape[0]
        states = np.copy(states)
        # states[:, 3:] -= np.pi  # Hack to treat angles properly
        # arm_bases = np.vstack((states[:, 0], states[:, 1] \
        # + self.base_length/2)).T # (B, 2)
        link_thetas = np.cumsum(states, axis=1)  # (B, num_links)
        link_cosines = np.cos(link_thetas)  # (B, num_links)
        link_sines = np.sin(link_thetas)  # (B, num_links)
        normalized_link_points = np.stack(
            (link_cosines, link_sines), axis=2
        )  # (B, num_links, 2)
        point_der = (
            self.arm_link_lengths.reshape(1, self.num_links, 1) * normalized_link_points
        )  # (B, num_links, 2)
        joint_pos = np.cumsum(point_der, axis=1)
        return np.concatenate((np.zeros((B, 1, 2)), joint_pos), axis=1)

    def batch_get_robot_representations(self, states) -> dict:
        segment_points = self.batch_forward_kinematics(states)
        start_points = segment_points[:, :-1, :]
        end_points = segment_points[:, 1:, :]
        segments = (
            np.concatenate((start_points, end_points), axis=2)
            .reshape(-1, 4)
            .reshape(-1, 2, 2)
        )
        return {
            "rectangles": np.empty((0, 4)),
            "segments": segments,
            "points": np.empty((0, 2)),
            "points_radius": 0.0,
            "segments_radii": 0.1,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = FixedArm()
    state = env.sample_point()

    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()
