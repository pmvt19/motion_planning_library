import numpy as np
from shapely import Polygon, Point, LineString
from state import NumpyState, AngularNumpyState
import matplotlib.pyplot as plt

class RobotSpace():
    def __init__(self):
        self.num_collision_checks = 0
        self.obstacles = []

    def is_valid(self, state):
        raise NotImplementedError
    
    def is_valid_edge(self, start, end):
        raise NotImplementedError
    
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
    
    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        for obs in self.obstacles:
            x,y = obs.exterior.xy
            ax.plot(x,y, color='black')
    
    def animate_path(self, path, frame_delay=0.1):
        for state in path:
            plt.clf()
            self.draw_environment(plt.gca())
            self.draw_state(plt.gca(), state)
            plt.pause(frame_delay)
        
    def get_state_value(self, state):
        if isinstance(state, NumpyState):
            return state.value
        elif isinstance(state, np.ndarray):
            return state
        else:
            raise ValueError("Incorrect Input Type")
    
class HolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()

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
        for obs in self.obstacles:
            if robot.within(obs):
                return False
        return True
    
    def get_edge_states(self, start : np.ndarray, end : np.ndarray):
        edge_length = np.linalg.norm(end - start)
        dir = (end - start) / edge_length
        
        num_checks = int(edge_length / self.edge_validity_delta)
        edge_states_derivative = np.tile(dir * self.edge_validity_delta, (num_checks+2, 1))
        edge_states_derivative[0] = np.zeros_like(start)
        edge_states = np.cumsum(edge_states_derivative, axis=0) + start
        edge_states[-1] = end
        return edge_states
    
    def is_valid_edge(self, start, end):
        start = self.get_state_value(start)
        end = self.get_state_value(end)

        edge_states = self.get_edge_states(start, end)

        for state in edge_states:
            if not self.is_valid(state):
                return False
        return True

    def shoot_ray(self, node, sampled_point, delta):
        if node == sampled_point:
            return node
        node = self.get_state_value(node)
        sampled_point = self.get_state_value(sampled_point)

        edge_length = self.dist(self.make_state(sampled_point), self.make_state(node))
        dir = (sampled_point - node) / edge_length
        extension_dist = np.random.uniform(low=0, high=delta)
        
        target_position = node + dir * extension_dist
        edge_states = self.get_edge_states(node, target_position)

        prev_state = node 
        for state in edge_states[1:]:
            if not self.is_valid(state):
                return self.make_state(prev_state)
            prev_state = state
        return self.make_state(prev_state)

    def draw_state(self, ax, state):
        robot = self.generate_robot_representation(state)
        ax.scatter(*robot.xy, color='red')

    def sample_task(self):
        start = self.sample_valid_point()
        target = self.sample_valid_point()
        return start, target

class NonHolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()
    
    def make_control(self, state : np.ndarray):
        raise NotImplementedError
