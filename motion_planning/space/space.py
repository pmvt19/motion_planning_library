import numpy as np
import time
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from shapely import Polygon, Point, LineString, affinity
from collections import defaultdict
from sklearn.metrics import pairwise_distances

from motion_planning.tools import NumpyState, AngularNumpyState
from motion_planning.obstacle_sets import ObstacleSet, ParkingSpace
from motion_planning.utils import create_rectangle_geometry, numpystate_distance, issue_warning, interpolate_edge, batch_interpolate_edge, batch_interpolate_edge_uniform, rad2deg
from motion_planning.controller.xbox_controller import XboxController

class RobotSpace():
    def __init__(self):
        self.num_collision_checks = 0
        self.obstacles = []
        self.edge_validity_delta = 0.5

        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.boundary = create_rectangle_geometry(x_loc=((self.x_range[0]+self.x_range[1])/2), 
                                                    y_loc=((self.y_range[0]+self.y_range[1])/2),
                                                    x_width=self.x_range[1]-self.x_range[0],
                                                    y_length=self.y_range[1]-self.y_range[0])
        
        self.angular_dims_start = None


        self.state_dims = None
        self.workspace_dims = None

    def is_valid(self, state):
        raise NotImplementedError
    
    def is_valid_edge(self, start, end):
        edge_states = interpolate_edge(start, end, self.edge_validity_delta)

        for state in edge_states:
            if not self.is_valid(state):
                return False
        return True
    
    def make_state(self, state):
        raise NotImplementedError
    
    def sample_point(self):
        raise NotImplementedError
    
    def dist(self, state1, state2):
        raise NotImplementedError
    
    def generate_robot_representation(self, state):
        raise NotImplementedError
    
    def sample_valid_point(self):
        point = self.sample_point()
        while not self.is_valid(point):
            point = self.sample_point()
        return point
    
    def draw_state(self, ax, state):
        raise NotImplementedError
    
    # def draw_state(self, ax, state):
    #     robot = self.generate_robot_representation(state)
    #     for rectangle in robot.rectangles:
    #         ax.plot(*rectangle.exterior.xy, color='red')
    #     for segment in robot.segments:
    #         ax.plot(*segment.xy, color='red')
    #     for point in robot.points:
    #         ax.plot(*point.xy, color='red')
    
    def draw_environment(self, ax):
        containers = []
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        for obs in self.obstacles:
            x,y = obs.exterior.xy
            containers.extend(ax.plot(x,y, color='black'))
        return containers
    
    def animate_path(self, path, frame_delay=0.1):
        for state in path:
            plt.clf()
            self.draw_environment(plt.gca())
            self.draw_state(plt.gca(), state)
            plt.pause(frame_delay)

    def animate_path_upgraded(self, path, frame_delay=100):
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        artists = []
        for state in path:
            frame_containers = []
            frame_containers.extend(self.draw_environment(ax))
            container = self.draw_state(ax, state)
            frame_containers.extend(container)
            artists.append(frame_containers)
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=frame_delay, repeat=False)
        plt.show()

    def get_state_value(self, state):
        if isinstance(state, NumpyState):
            return state.value
        elif isinstance(state, np.ndarray):
            return state
        else:
            raise ValueError("Incorrect Input Type")

    def shoot_ray(self, node, sampled_point, delta):
        if node == sampled_point:
            return node
        node = self.get_state_value(node)
        sampled_point = self.get_state_value(sampled_point)

        edge_length = self.dist(self.make_state(sampled_point), self.make_state(node))
        dir = (sampled_point - node) / edge_length
        extension_dist = np.random.uniform(low=0, high=delta)
        
        target_position = node + dir * extension_dist
        edge_states = interpolate_edge(self.make_state(node), self.make_state(target_position), self.edge_validity_delta)
        prev_state = node 
        for state in edge_states[1:]:
            if not self.is_valid(state):
                return self.make_state(prev_state)
            prev_state = state
        return self.make_state(prev_state)
    
    def set_obstacles(self, obstacle_set : ObstacleSet):
        self.obstacles = obstacle_set.obstacles
        self.boundary = obstacle_set.boundary

        x_points, y_points = self.boundary.exterior.xy
        self.x_range = [min(x_points), max(x_points)]
        self.y_range = [min(y_points), max(y_points)]
    
    def batch_is_valid(self, states : np.ndarray):
        validities = []
        for state in states:
            validities.append(self.is_valid(self.make_state(state)))
        validities = np.array(validities)
        return validities
    
    def batch_is_valid_edge(self, start_states : np.ndarray, end_states : np.ndarray):
        B, d = start_states.shape
        # start_states: (B, d), end_states: (B, d)
        pts, steps = batch_interpolate_edge(start_states, end_states, self.edge_validity_delta, self.angular_dims_start)
        pts = pts.reshape(-1, d)
        pt_validities = self.batch_is_valid(pts).reshape(B, -1)
        edge_validities = np.array([np.all(pt_validities[i, :steps[i]]) for i in range(len(steps))])
        return edge_validities

    def batch_is_valid_edge_uniform(self, start_states : np.ndarray, end_states : np.ndarray):
        B, d = start_states.shape
        # start_states: (B, d), end_states: (B, d)
        pts = batch_interpolate_edge_uniform(start_states, end_states, self.edge_validity_delta, self.angular_dims_start)
        pts = pts.reshape(-1, d)
        pt_validities = self.batch_is_valid(pts).reshape(B, -1)
        edge_validities = np.all(pt_validities, axis=1)
        return edge_validities
    
    def batch_sample_point(self, num_points):
        raise NotImplementedError
        
    def batch_get_robot_representations(self, states : np.ndarray):
        raise NotImplementedError
    
    def batch_sample_points_around_target(self, targets: np.ndarray):
        raise NotImplementedError