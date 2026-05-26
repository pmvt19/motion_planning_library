import numpy as np
from shapely import affinity

from motion_planning.controller.xbox_controller import XboxController
from motion_planning.space import NonHolonomicRobot
from motion_planning.tools import AngularNumpyState, NumpyState
from motion_planning.utils import create_rectangle_geometry, numpystate_distance


class SkidSteerCar(NonHolonomicRobot):
    def __init__(self, edge_validity_delta: float = 0.05):
        super().__init__()

        self.edge_validity_delta = edge_validity_delta

        self.car_width = 1
        self.car_length = 2

        self.angular_dim_start = 2

        self.x_range = [-10, 10]
        self.y_range = [-10, 10]
        self.theta_range = [0, 2 * np.pi]

        self.velocity_range = [-3, 3]
        self.delta_range = [-np.pi, np.pi]

        self.state_dim = 3
        self.control_dim = 2

    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        theta = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        return self.make_state(np.array([x, y, theta]))

    def sample_controls(self, bias=0.5):
        v = np.random.uniform(
            low=self.velocity_range[0], high=self.velocity_range[1]
        )  # Sample Velocity Uniformly Between: (-3, 3)
        delta = np.random.uniform(
            low=self.delta_range[0], high=self.delta_range[1]
        )  # Sample Delta Uniformly Between: (-pi, pi)

        # Can only Move Forward OR Turn in place Not Both!
        if np.random.random() < bias:
            delta = 0.0
        else:
            v = 0.0
        return self.make_control(np.array([v, delta]))

    def make_state(self, state: np.ndarray):
        return AngularNumpyState(state, angular_dims_start=self.angular_dim_start)

    def make_control(self, control: np.ndarray):
        return NumpyState(control)

    def clip_state(self, state: np.ndarray):
        state = np.clip(
            state,
            np.array([self.x_range[0], self.y_range[0], -np.inf]),
            np.array([self.x_range[1], self.y_range[1], np.inf]),
        )
        return state

    def dist(self, state1, state2):
        state1 = self.get_state_value(state1)
        state2 = self.get_state_value(state2)
        return numpystate_distance(self.make_state(state1), self.make_state(state2))

    def is_valid(self, state):
        self.num_collision_checks += 1
        robot = self.generate_robot_representation(state)
        for obs in self.obstacles:
            if obs.intersects(robot):
                return False
        return True

    def generate_robot_representation(self, state):
        x, y, theta = self.get_state_value(state)
        theta -= np.pi / 2
        robot = create_rectangle_geometry(
            x_loc=x, y_loc=y, x_width=self.car_width, y_length=self.car_length
        )
        robot = affinity.rotate(robot, theta, use_radians=True)
        return robot

    def generate_costmetic_robot_representation(self, state):
        print("Cosmetic State is Incomplete")
        return self.generate_robot_representation(state), []

    def draw_state(self, ax, state):
        robot, cosmetics = self.generate_costmetic_robot_representation(state)
        x, y = robot.exterior.xy
        ax.plot(x, y, color="red")

        for c in cosmetics:
            x, y = c.exterior.xy
            ax.plot(x, y, color="black")

    def state_derivative(self, state, control):
        x, y, theta = self.get_state_value(state)
        v, delta = self.get_state_value(control)
        x_dot = np.array(
            [
                v * np.cos(theta) * self.dt,
                v * np.sin(theta) * self.dt,
                delta * self.dt,
            ]
        )
        return x_dot

    def input_to_control(self, inputs):
        # Inputs : [0, 0, 0, 0]
        # left, right, up, down =
        # down, left, right, up = inputs
        left = inputs[XboxController.XboxControls.X]
        right = inputs[XboxController.XboxControls.B]
        up = inputs[XboxController.XboxControls.Y]
        down = inputs[XboxController.XboxControls.A]
        if left:
            return self.make_control(np.array([0.0, self.delta_range[0]]))
        if right:
            return self.make_control(np.array([0.0, self.delta_range[1]]))
        if up:
            return self.make_control(np.array([self.velocity_range[0], 0.0]))
        if down:
            return self.make_control(np.array([self.velocity_range[1], 0.0]))
        return self.make_control(np.array([0.0, 0.0]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = SkidSteerCar()
    state = env.sample_point()

    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()
