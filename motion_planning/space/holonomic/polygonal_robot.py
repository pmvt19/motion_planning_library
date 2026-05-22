import numpy as np

from shapely import affinity

from motion_planning.space import HolonomicRobot
from motion_planning.tools import AngularNumpyState
from motion_planning.utils import create_rectangle_geometry, numpystate_distance
from motion_planning.controller.xbox_controller import XboxController

class PolygonalRobot(HolonomicRobot):
    def __init__(self, edge_validity_delta: float = 0.5, robot_width: float = 0.5, robot_length: float = 3):
        super().__init__()

        self.edge_validity_delta = edge_validity_delta

        self.x_range = [-10,10]
        self.y_range = [-10,10]

        self.theta_range = [0, 2*np.pi]

        self.angular_dims_start = 2

        self.robot_width = robot_width
        self.robot_length = robot_length

        self.obstacles = []

        self.do_boundary_check = True

    def make_state(self, state : np.ndarray):
        return AngularNumpyState(value=state, angular_dims_start=self.angular_dims_start)

    def generate_robot_representation(self, state):
        x, y, theta = self.get_state_value(state)
        robot = create_rectangle_geometry(x_loc=x, 
                                          y_loc=y, 
                                          x_width=self.robot_width, 
                                          y_length=self.robot_length)
        robot = affinity.rotate(robot, angle=theta, use_radians=True, origin=(x,y))
        return robot
    
    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        theta = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        return self.make_state(np.array([x, y, theta]))
    
    def dist(self, state1 : AngularNumpyState, state2 : AngularNumpyState):
        return numpystate_distance(state1, state2)
    
    def is_valid(self, state):
        self.num_collision_checks += 1
        robot = self.generate_robot_representation(state)

        if self.do_boundary_check and (not robot.within(self.boundary)):
            return False
        
        for obs in self.obstacles:
            if obs.intersects(robot):
                return False
        return True

    def draw_state(self, ax, state):
        robot = self.generate_robot_representation(state)
        ax.plot(*robot.exterior.xy, color='red')
    
    def input_to_x_dot(self, inputs) -> np.ndarray:
        dt = 0.1
        x_dot = inputs[XboxController.XboxControls.LTHUMBX] * dt
        y_dot = -inputs[XboxController.XboxControls.LTHUMBY] * dt
        theta_dot = inputs[XboxController.XboxControls.RTHUMBX] * dt
        return np.array([x_dot, y_dot, theta_dot])


    ### Batch Methods ###
    
    def batch_get_robot_representations(self, states: np.ndarray):
        # states : (B, 3)
        B, d = states.shape

        # Extract Parameter Values
        xs = states[:, 0]
        ys = states[:, 1]
        thetas = states[:, 2]

        # Tranformation Matrices
        # tmats : (B, 2, 3)

        cosines = np.cos(thetas)
        sines = np.sin(thetas)

        tmats = np.array([[cosines, -sines, xs],
                          [sines, cosines, ys]])
        tmats = tmats.transpose(2, 0, 1)

        # end_points : (B, 2, 3) Points will be in Homogenous Coordinates
        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)

        end_points1 = np.concatenate((np.zeros((B,1)), np.zeros((B,1)) + self.robot_length/2, np.ones((B,1))), axis=1)
        end_points2 = np.concatenate((np.zeros((B,1)), np.zeros((B,1)) - self.robot_length/2, np.ones((B,1))), axis=1)
        end_points = np.concatenate((end_points1.reshape(B, 1, -1), end_points2.reshape(B, 1, -1)), axis=1)

        segments = tmats @ end_points.transpose(0,2,1)
        segments = segments.reshape(B, 2, 2).transpose(0,2,1)

        radii = np.ones((B, 1)) * self.robot_width/2
        
        # TODO: Standardize the returnable for this function
        return {
            'rectangles' : np.empty((0, 4)),
            'segments' : segments, 
            'points' : np.empty((0, 2)),
            'points_radius': 0.0,
            'segments_radii' : radii, 
            'line_approx_non_aarect' : segments,
            'line_approx_non_aarect_radii' : radii,
        }
    
    def batch_sample_points_around_target(self, targets: np.ndarray):
        B, _ = targets.shape

        num_thetas_per_target = 20

        thetas = np.random.uniform(low=0, high=2*np.pi, size=(B*num_thetas_per_target,1))
        points = np.repeat(targets, num_thetas_per_target, axis=0)

        points = np.concatenate((points,thetas), axis=1)
        validities = self.batch_is_valid(points)
        return points[validities]
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = PolygonalRobot()
    state = env.sample_point()

    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()