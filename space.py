import numpy as np
from shapely import Polygon, Point, LineString, affinity
from state import NumpyState, AngularNumpyState
import matplotlib.pyplot as plt
from utils import create_rectangle_geometry, numpystate_distance, interpolate_SE2_edge

class RobotSpace():
    def __init__(self):
        self.num_collision_checks = 0
        self.obstacles = []


        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.boundary = create_rectangle_geometry(x_loc=((self.x_range[0]+self.x_range[1])/2), 
                                                    y_loc=((self.y_range[0]+self.y_range[1])/2),
                                                    x_width=self.x_range[1]-self.x_range[0],
                                                    y_length=self.y_range[1]-self.y_range[0])

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
    
    def sample_task(self):
        start = self.sample_valid_point()
        target = self.sample_valid_point()
        return start, target

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

class PolygonalRobot(HolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.5

        self.x_range = [-10,10]
        self.y_range = [-10,10]

        self.theta_range = [0, 2*np.pi]

        self.angular_dims_start = 2

        self.robot_width = 1
        self.robot_length = 2

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
    
    def get_edge_states(self, start : np.ndarray, end : np.ndarray):
        edge_states = interpolate_SE2_edge(start, end, self.edge_validity_delta)
        edge_states[:, 2] = edge_states[:, 2] % (2*np.pi)
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
        ax.plot(*robot.exterior.xy, color='red')
    


class NonHolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()
    
    def make_control(self, state : np.ndarray):
        raise NotImplementedError

if __name__ == '__main__':
    np.random.seed(0)
    env = PolygonalRobot()
    start, target = env.sample_task()
    print(start.value, target.value)
    env.get_edge_states(start.value, target.value)

    pointrobot = PointRobot()
    print(pointrobot.get_edge_states(start.value, target.value))
    # env.draw_environment(plt.gca())
    # env.draw_state(plt.gca(), start)
    # env.draw_state(plt.gca(), target)
    # plt.show()
