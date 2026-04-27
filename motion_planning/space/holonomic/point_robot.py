import numpy as np

from shapely import Point

from motion_planning.space import HolonomicRobot
from motion_planning.tools import NumpyState

class PointRobot(HolonomicRobot):
    def __init__(self):
        super().__init__()
        self.edge_validity_delta = 0.5

        self.x_range = [-10,10]
        self.y_range = [-10,10]

    def make_state(self, state):
        return NumpyState(state)

    def generate_robot_representation(self, state):
        state = self.get_state_value(state)
        robot = Point(state)
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
        robot = self.generate_robot_representation(state)
        return [ax.scatter(*robot.xy, color='red')]

    ## ---- Batched Methods ---- ##

    def batch_get_robot_representations(self, states : np.ndarray):
        # states # (B, 2)
        return {
            'rectangles' : np.empty((0, 4)),
            'segments' : np.empty((0, 2, 2)), 
            'segments_radii' : 0.0, 
            'points' : states,
            'points_radius': 0.0
        }
    
    def batch_sample_points_around_target(self, targets):
        validities = self.batch_is_valid(targets)
        return targets[validities]
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = PointRobot()
    state = env.sample_point()

    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()

    