import numpy as np

from shapely import Point

from motion_planning.space.holonomic.holonomic_robot import HolonomicRobot
from motion_planning.tools import NumpyState
from motion_planning.controller.xbox_controller import XboxController

class DiscRobot(HolonomicRobot):
    def __init__(self, edge_validity_delta: float = 0.5, disc_radius: float = 1.5):
        super().__init__()
        self.edge_validity_delta = edge_validity_delta
        self.disc_radius = disc_radius

        self.x_range = [-10,10]
        self.y_range = [-10,10]

    def make_state(self, state):
        return NumpyState(state)

    def generate_robot_representation(self, state):
        # TODO
        state = self.get_state_value(state)
        robot = Point(state).buffer(self.disc_radius)
        return robot
    
    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        return self.make_state(np.array([x, y]))
    
    def dist(self, state1, state2):
        return np.linalg.norm(self.get_state_value(state1) - self.get_state_value(state2))
    
    def is_valid(self, state):
        self.num_collision_checks += 1
        robot = self.generate_robot_representation(state)

        if not robot.within(self.boundary):
            return False

        for obs in self.obstacles:
            # if robot.within(obs):
            if obs.intersects(robot):
                return False
        return True

    def draw_state(self, ax, state):
        # TODO:
        robot = self.generate_robot_representation(state)
        xr, yr = robot.exterior.xy
        ax.fill(xr, yr, color='red')
    
    def input_to_x_dot(self, inputs): 
        dt = 0.1
        x_dot = inputs[XboxController.XboxControls.LTHUMBX] * dt
        y_dot = -inputs[XboxController.XboxControls.LTHUMBY] * dt
        return np.array([x_dot, y_dot])

    ## ---- Batched Methods ---- ##

    def batch_get_robot_representations(self, states: np.ndarray):
        # TODO:
        # states # (B, 2)
        return {
            'rectangles' : np.empty((0, 4)),
            'segments' : np.empty((0, 2, 2)), 
            'segments_radii' : 0.0, 
            'points' : states,
            'points_radius': self.disc_radius
        }
    
    def batch_sample_points_around_target(self, targets: np.ndarray):
        validities = self.batch_is_valid(targets)
        return targets[validities]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    env = DiscRobot()
    state = env.sample_point()

    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()