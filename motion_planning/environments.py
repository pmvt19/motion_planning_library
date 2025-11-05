import numpy as np 
from shapely import Polygon, Point, LineString
import matplotlib.pyplot as plt
from collections import defaultdict
from state import NumpyState, AngularNumpyState
from shapely import affinity
from utils import interpolate_euclidean_edge, interpolate_angular_edge, interpolate_edge, numpystate_distance


class Environment():
    def __init__(self):
        self.num_collision_checks = 0

    def is_valid(self, point):
        raise NotImplementedError

    def is_valid_edge(self, start, end):
        raise NotImplementedError
    
    def sample_point(self):
        raise NotImplementedError
    
    def sample_valid_point(self):
        point = self.sample_point()
        while not self.is_valid(point):
            point = self.sample_point()
        return point
    
    def sample_task(self):
        start = self.sample_valid_point()
        target = self.sample_valid_point()
        return start, target
    
    def instance_check(self, start, end):
        if isinstance(start, NumpyState) and isinstance(end, NumpyState):
            start = start.value
            end = end.value
            return start, end 
        elif isinstance(start, np.ndarray) and isinstance(end, np.ndarray):
            return start, end 
        else:
            print(f"Mismatched Input Types: {type(start)}, {type(end)}")
            raise NotImplementedError
    
    def dist(self, state1, state2):
        raise NotImplementedError
    
    def create_rectangle_geometry(self, x_loc, y_loc, x_width, y_length):
        shape = Polygon([[x_loc-x_width/2, y_loc-y_length/2], 
                         [x_loc-x_width/2, y_loc+y_length/2],
                         [x_loc+x_width/2, y_loc+y_length/2],
                         [x_loc+x_width/2, y_loc-y_length/2],])
        return shape

class Workspace2dEnv(Environment):
    def __init__(self):
        super().__init__()
        self.obstacles = []
        

        self.edge_validity_delta = 0.5
        self.interpolated_edge_points = defaultdict(list)

    def dist(self, point1, point2):
        return np.linalg.norm(point1-point2)


    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        for obs in self.obstacles:
            x,y = obs.exterior.xy
            ax.plot(x,y, color='red')

    def is_valid(self, point):
        self.num_collision_checks += 1
        if isinstance(point, NumpyState):
            point = Point(point.value)
        elif isinstance(point, np.ndarray):
            point = Point(point)
        else:
            raise NotImplementedError
        
        for obs in self.obstacles:
            if obs.contains(point) or obs.boundary.contains(point):
                return False
        return True
    
    def make_node_pair_key(self, node1, node2):
        assert (isinstance(node1, np.ndarray))
        assert (isinstance(node2, np.ndarray))
        return tuple(np.hstack((node1, node2)))
    
    def is_valid_edge(self, start, end):
        # start = np.array(start)
        # end = np.array(end)
        if isinstance(start, NumpyState) and isinstance(end, NumpyState):
            start = start.value
            end = end.value
        elif isinstance(start, np.ndarray) and isinstance(end, np.ndarray):
            pass
        else:
            print(f"Mismatched Input Types: {type(start)}, {type(end)}")
            raise NotImplementedError
        key = self.make_node_pair_key(start, end)
        
        edge_length = np.linalg.norm(end - start)
        dir = (end - start) / np.linalg.norm(end - start)

        cur_node = start
        num_checks = int(edge_length / self.edge_validity_delta)
        # self.interpolated_edge_points[key].append(cur_node)
        for i in range(num_checks):
            self.interpolated_edge_points[key].append(cur_node)
            if not self.is_valid(cur_node):
                return False
            cur_node = cur_node + dir * self.edge_validity_delta
        
        if self.is_valid(end):
            self.interpolated_edge_points[key].append(end)
        else:
            return False

        return True
    
    def shoot_ray(self, node, sampled_point, delta):
        if node == sampled_point:
            return node
        dir = (sampled_point.value - node.value) / np.linalg.norm(sampled_point.value - node.value)
        ext_amount = np.random.random() * delta
        new_node = node.value + (ext_amount * dir)
        new_node = self.make_state(new_node)
        while not (self.is_valid(new_node) and self.is_valid_edge(node, new_node)):
            ext_amount -= self.edge_validity_delta
            if ext_amount <= 0:
                return node
            new_node = node.value + (ext_amount * dir)
            new_node = self.make_state(new_node)
        return new_node

    # def shoot_ray(self, node, sampled_point, delta=0.5):
    #     dir = (np.array(sampled_point) - np.array(node)) / np.linalg.norm(np.array(sampled_point) - np.array(node))
    #     ext_amount = np.random.random() * delta
    #     new_node = node + (ext_amount * dir)
    #     key = self.make_node_pair_key(np.array(node), np.array(new_node))
    #     if self.is_valid_edge(node, new_node):
    #         new_node = self.interpolated_edge_points[key][-1]
    #     else:
    #         new_node = node
    #     return new_node

    def make_state(self, state : np.ndarray):
        return NumpyState(state)

    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        for obs in self.obstacles:
            x,y = obs.exterior.xy
            ax.plot(x,y, color='red')
    
    def sample_point(self):
        rand_point = np.random.random(size=2)
        rand_point[0] = (np.random.random() * (self.x_range[1] - self.x_range[0])) + self.x_range[0]
        rand_point[1] = (np.random.random() * (self.y_range[1] - self.y_range[0])) + self.y_range[0]
        return self.make_state(rand_point)

class OpenSpace2d(Workspace2dEnv):
    def __init__(self):
        super().__init__()
        self.x_range = [-10,10]
        self.y_range = [-10,10]

class Environment2d(Workspace2dEnv):    
    def __init__(self):
        super().__init__()
        self.obstacles = [
            Polygon([
                [6, 6],
                [7, 6],
                [7, 7],
                [6, 7],            
            ]),
            Polygon([
                [2.5, -7.5],
                [5, -7.5],
                [5, 7.5],
                [2.5, 7.5],
            ])
        ]

        self.x_range = [-10,10]
        self.y_range = [-10,10]

        self.env_boundaries = []
        for x in self.x_range:
            for y in self.y_range:
                self.env_boundaries.append([x, y])
        
class RandomSamplePassage(Workspace2dEnv):
    def __init__(self, num_walls=3):
        super().__init__()
        self.obstacles = []

        self.x_range = [0,(10 * (num_walls+1))]
        self.y_range = [0,10]

        self.env_boundaries = []
        for x in self.x_range:
            for y in self.y_range:
                self.env_boundaries.append([x, y])

        wall_width = 1
        gap_width = 1
        
        for i in range(num_walls):
            x_low = (10 * (i+1)) - wall_width/2
            x_high = (10 * (i+1)) + wall_width/2
            gap_y_loc = np.random.random() * (self.y_range[1] - self.y_range[0] - gap_width) + self.y_range[0] + gap_width/2

            y_low = gap_y_loc - gap_width/2
            y_high = gap_y_loc + gap_width/2
            
            obs = Polygon([[x_low, self.y_range[0]],
                           [x_low, y_low],
                           [x_high, y_low],
                           [x_high, self.y_range[0]]])
            self.obstacles.append(obs)
            
            obs = Polygon([[x_low, self.y_range[1]],
                           [x_low, y_high],
                           [x_high, y_high],
                           [x_high, self.y_range[1]]])
            self.obstacles.append(obs)

class NonholonomicEnv(Environment):
    def __init__(self):
        super().__init__()

    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        for obs in self.obstacles:
            x,y = obs.exterior.xy
            ax.plot(x,y, color='black')
    
    # def draw_state(self, ax, state : NumpyState):
    #     robot = self.generate_robot_representation(state)
    #     x,y = robot.exterior.xy
    #     ax.plot(x,y, color='red')
    #     xs, ys, ts = state.value

    def draw_state(self, ax, state : NumpyState):
        robot, cosmetics = self.generate_costmetic_robot_representation(state)
        x,y = robot.exterior.xy
        ax.plot(x,y, color='red')

        for c in cosmetics:
            x,y = c.exterior.xy
            ax.plot(x,y, color='black')

    def animate_path(self, path, frame_delay=0.1):
        for state in path:
            plt.clf()
            self.draw_environment(plt.gca())
            self.draw_state(plt.gca(), state)
            plt.pause(frame_delay)
    
    def simulate(self, starting_state: AngularNumpyState, control_seq: list):
        state = starting_state
        state_seqs = [state]
        for control, time in control_seq:
            state, _, _ = self.extend_state(state, time, control, do_collision_checking=False)
            state_seqs.append(state)
        return state_seqs
    
    def extend_state(self, state: AngularNumpyState, time: float, controls=None, do_collision_checking=True):
        if controls is None:
            controls = self.sample_controls()
        
        num_iterations = int(time / self.dt)
        # print(num_iterations, time, self.dt, time/self.dt, int(time/self.dt), time//self.dt)
        # assert(int(time/self.dt) == time//self.dt)
        list_of_states = [state]
        running_time = 0
        
        for i in range(num_iterations):
            state = self.simulate_forward(state, controls)
            if do_collision_checking and not self.is_valid(state):
                break
            # running_time += self.dt
            running_time = (i+1) * self.dt
            list_of_states.append(state)

        return list_of_states[-1], controls, running_time

class CarParkingEnv(NonholonomicEnv):
    def __init__(self):
        super().__init__()

        self.obstacles = []
        self.parking_space_samples = []
        self.parking_space_centers = []

        self.x_range = [-15,15]
        self.y_range = [-15,15]
        self.angular_dims_start = 2

        # self.obstacles.extend(self.create_parking_space())
        # self.obstacles.extend(self.create_parking_space(x_loc=-5, y_loc=-5))

        self.obstacles.extend(self.create_parking_space(space_width=5))
        self.obstacles.extend(self.create_parking_space(x_loc=-7.5, y_loc=-7.5, space_width=5))

        # a car state is x,y,theta
        

        self.edge_validity_delta = 0.5

        self.car_width = 2
        self.car_length = 4

        self.dt = 0.01

        self.state_dim = 3
        self.control_dim = 2
    
    def create_parking_space(self, x_loc=0, y_loc=0, space_width=3):
        # space_width = 3 
        # space_width = 5
        line_width = 0.5
        line_height = 6
        obs = [
            Polygon([
                [x_loc, y_loc],
                [x_loc, y_loc+line_height],
                [x_loc+line_width, y_loc+line_height],
                [x_loc+line_width, y_loc],            
            ]),
            Polygon([
                [x_loc+space_width+line_width, y_loc],
                [x_loc+space_width+line_width, y_loc+line_height],
                [x_loc+space_width+line_width*2, y_loc+line_height],
                [x_loc+space_width+line_width*2, y_loc],            
            ]),
            Polygon([ # Horizontal Bar
                [x_loc+line_width, y_loc+line_height-line_width],
                [x_loc+line_width, y_loc+line_height],
                [x_loc+space_width+line_width, y_loc+line_height],
                [x_loc+space_width+line_width, y_loc+line_height-line_width],            
            ]),
        ]
        x_center = (2 * x_loc + space_width + line_width*2) / 2
        y_center = (2 * y_loc + line_height - line_width) / 2

        self.parking_space_centers.append(self.make_state(np.array([x_center, y_center, 0.0])))

        sample_radius = 2
        space_samples = np.array([x_center, y_center, 0]) + (np.random.normal(size=(1000, 3)) * sample_radius)
        self.parking_space_samples.extend(space_samples)

        return obs
    
    def generate_robot_representation(self, state):
        if isinstance(state, AngularNumpyState):
            x, y, theta = state.value
        elif isinstance(state, np.ndarray):
            x, y, theta = state
        else:
            raise ValueError("state input type is invalid")
        
        robot = Polygon([
                    [x-self.car_width/2, y-self.car_length/2],
                    [x-self.car_width/2, y+self.car_length/2],
                    [x+self.car_width/2, y+self.car_length/2],
                    [x+self.car_width/2, y-self.car_length/2],
                ])
        rotated_robot = affinity.rotate(robot, theta, use_radians=True)
        return rotated_robot
    
    def generate_costmetic_robot_representation(self, state):
        if isinstance(state, AngularNumpyState):
            x, y, theta = state.value
        elif isinstance(state, np.ndarray):
            x, y, theta = state
        else:
            raise ValueError("state input type is invalid")
        
        robot = self.generate_robot_representation(state)
        robot_centroid = robot.centroid

        self.x_offset = 0.1
        self.y_offset = 0.3
        self.wheel_length = 0.4
        self.wheel_width = 0.1

        fr_wheel = self.create_rectangle_geometry(x_loc=x+self.car_width/2+self.x_offset,
                                                  y_loc=y+self.car_length/2-self.y_offset,
                                                  x_width=self.wheel_width,
                                                  y_length=self.wheel_length)
        fl_wheel = self.create_rectangle_geometry(x_loc=x-self.car_width/2-self.x_offset,
                                                  y_loc=y+self.car_length/2-self.y_offset,
                                                  x_width=self.wheel_width,
                                                  y_length=self.wheel_length)
        fr_wheel = affinity.rotate(fr_wheel, theta, use_radians=True, origin=robot_centroid)
        fl_wheel = affinity.rotate(fl_wheel, theta, use_radians=True, origin=robot_centroid)

        return robot, [fr_wheel, fl_wheel]

    def make_state(self, state : np.ndarray):
        # State is X, Y, Theta
        return AngularNumpyState(state, angular_dims_start=self.angular_dims_start)
    
    def make_control(self, control : np.ndarray):
        # State is V, Delta
        return NumpyState(control)

    def sample_point(self):
        rand_point = np.random.random(size=3)
        rand_point[0] = (np.random.random() * (self.x_range[1] - self.x_range[0])) + self.x_range[0]
        rand_point[1] = (np.random.random() * (self.y_range[1] - self.y_range[0])) + self.y_range[0]
        rand_point[2] = (np.random.random() * (2*np.pi))
        return self.make_state(rand_point)
    
    def is_valid(self, state):
        self.num_collision_checks += 1
        robot = self.generate_robot_representation(state)
        for obs in self.obstacles:
            if obs.intersects(robot):
                return False
        return True
    
    def is_valid_edge(self, start, end):
        start, end = self.instance_check(start, end)
        interpolated_points = interpolate_edge(start, end, delta=self.edge_validity_delta)

        for point in interpolated_points:
            if not self.is_valid(point):
                return False
        return True
    
    def shoot_ray(self, node, sampled_point, delta):
        dir = (sampled_point.value - node.value) / np.linalg.norm(sampled_point.value - node.value)
        ext_amount = np.random.random() * delta
        new_node = node.value + (ext_amount * dir)
        new_node = self.make_state(new_node)
        while not (self.is_valid(new_node) and self.is_valid_edge(node, new_node)):
            ext_amount -= self.edge_validity_delta
            if ext_amount <= 0:
                return node
            new_node = node.value + (ext_amount * dir)
            new_node = self.make_state(new_node)
        return new_node
    
    def dist(self, state1, state2):
        return numpystate_distance(self.make_state(state1), self.make_state(state2))

    def sample_controls(self, dom_bias=0.5):
        v = (np.random.random() * 6) - 3 # Sample Velocity Uniformly Between: (-3, 3)
        delta = (np.random.random() * 2*np.pi) - np.pi # Sample Delta Uniformly Between: (-pi, pi)
        # print(v, delta)

        # Can only Move Forward OR Turn in place Not Both!
        if np.random.random() < dom_bias:
            delta = 0.0
        else:
            v = 0.0
        return self.make_control(np.array([v, delta]))
    
    def simulate_forward(self, state: AngularNumpyState, controls : NumpyState):
        v, delta = controls.value
        x, y, theta = state.value
        theta -= np.pi/2 # Hack to treat the upward direction as the 0 radians orientation (Should Fix)
        
        x_dot = np.array([
                    v * np.cos(theta) * self.dt,
                    v * np.sin(theta) * self.dt,
                    delta * self.dt,
                ])
        new_state = np.copy(state.value) + x_dot
        return self.make_state(new_state)
    
    def get_fixed_task(self):
        idxes = np.random.choice(len(self.parking_space_centers), size=(2,), replace=False)
        start = self.parking_space_centers[idxes[0]]
        target = self.parking_space_centers[idxes[1]]
        return start, target

class DubinsCarEnv(NonholonomicEnv):
    def __init__(self):
        super().__init__()
        self.obstacles = []
        self.obstacles.append(
            self.create_rectangle_geometry(x_loc=0, y_loc=0, x_width=4, y_length=4)
        )

        self.x_range = [-15,15]
        self.y_range = [-15,15]

        self.boundary = self.create_rectangle_geometry(x_loc=((self.x_range[0]+self.x_range[1])/2), 
                                                       y_loc=((self.y_range[0]+self.y_range[1])/2),
                                                       x_width=self.x_range[1]-self.x_range[0],
                                                       y_length=self.y_range[1]-self.y_range[0])

        self.velocity_range = [-3, 3]
        self.theta_range = [0, 2*np.pi]
        self.phi_range = [-np.pi/3, np.pi/3]

        self.accel_range = [-5, 5]
        self.psi_range = [-1, 1]

        self.edge_validity_delta = 0.5

        self.angular_dims_start = 4

        self.car_width = 2
        self.car_length = 4

        self.dt = 0.05

        self.state_dim = 5
        self.control_dim = 2

    def make_state(self, state : np.ndarray):
        # State is X, Y, V, Phi, Theta
        return AngularNumpyState(state, angular_dims_start=self.angular_dims_start)
    
    def make_control(self, control : np.ndarray):
        # State is A, Psi
        return NumpyState(control)
    
    def sample_point(self):
        rand_point = np.zeros(5)
        rand_point[0] = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        rand_point[1] = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        rand_point[2] = np.random.uniform(low=self.velocity_range[0], high=self.velocity_range[1])
        # rand_point[3] = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        # rand_point[4] = np.random.uniform(low=self.phi_range[0], high=self.phi_range[1])

        rand_point[3] = np.random.uniform(low=self.phi_range[0], high=self.phi_range[1])
        rand_point[4] = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        return self.make_state(rand_point)
    
    def sample_controls(self):
        a = np.random.uniform(low=self.accel_range[0], high=self.accel_range[1])
        psi = np.random.uniform(low=self.psi_range[0], high=self.psi_range[1])
        return self.make_control(np.array([a, psi]))
    
    def simulate_forward(self, state: AngularNumpyState, controls : NumpyState):
        x, y, v, phi, theta = state.value
        a, psi = controls.value

        theta -= np.pi/2 # Hack to treat the upward direction as the 0 radians orientation (Should Fix)

        x_dot = np.array([
                    v * np.cos(theta) * self.dt,
                    v * np.sin(theta) * self.dt,
                    a * self.dt,
                    psi * self.dt,
                    v / self.car_length * np.tan(phi) * self.dt,
                ])
        
        new_state = np.copy(state.value) + self.make_state(x_dot).value
        return self.make_state(new_state)
    
    def decompose_state(self, state):
        if isinstance(state, AngularNumpyState):
            x, y, v, phi, theta  = state.value
        elif isinstance(state, np.ndarray):
            x, y, v, phi, theta = state
        else:
            raise ValueError("state input type is invalid")

        return x, y, v, phi, theta
    
    def is_valid_state_constraints(self, state):
        x, y, v, phi, theta = self.decompose_state(state)
        if v < self.velocity_range[0] or v > self.velocity_range[1]:
            return False
        if phi < self.phi_range[0] or phi > self.phi_range[1]:
            return False
        return True
    
    def dist(self, state1, state2):
        return numpystate_distance(self.make_state(state1), self.make_state(state2))
    
    def is_within_boundary(self, state):
        robot = self.generate_robot_representation(state)
        return robot.within(self.boundary)

    def is_valid(self, state):
        if self.is_valid_state_constraints(state) and self.is_within_boundary(state):
            self.num_collision_checks += 1
            robot = self.generate_robot_representation(state)
            for obs in self.obstacles:
                if obs.intersects(robot):
                    return False
            return True
        else:
            return False
    
    def is_valid_edge(self, start, end):
        start, end = self.instance_check(start, end)
        interpolated_points = interpolate_edge(start, end, delta=self.edge_validity_delta)

        for point in interpolated_points:
            if not self.is_valid(point):
                return False
        return True
    
    def generate_robot_representation(self, state):
        x, y, v, phi, theta = self.decompose_state(state)
        
        robot = Polygon([
                    [x-self.car_width/2, y-self.car_length/2],
                    [x-self.car_width/2, y+self.car_length/2],
                    [x+self.car_width/2, y+self.car_length/2],
                    [x+self.car_width/2, y-self.car_length/2],
                ])
        rotated_robot = affinity.rotate(robot, theta, use_radians=True)
        return rotated_robot
    
    def generate_costmetic_robot_representation(self, state):
        x, y, v, phi, theta = self.decompose_state(state)
        
        robot = self.generate_robot_representation(state)
        robot_centroid = robot.centroid

        self.x_offset = 0.1
        self.y_offset = 0.3
        self.wheel_length = 0.4
        self.wheel_width = 0.1

        fr_wheel = self.create_rectangle_geometry(x_loc=x+self.car_width/2+self.x_offset,
                                                  y_loc=y+self.car_length/2-self.y_offset,
                                                  x_width=self.wheel_width,
                                                  y_length=self.wheel_length)
        fl_wheel = self.create_rectangle_geometry(x_loc=x-self.car_width/2-self.x_offset,
                                                  y_loc=y+self.car_length/2-self.y_offset,
                                                  x_width=self.wheel_width,
                                                  y_length=self.wheel_length)
        fr_wheel = affinity.rotate(fr_wheel, theta, use_radians=True, origin=robot_centroid)
        fl_wheel = affinity.rotate(fl_wheel, theta, use_radians=True, origin=robot_centroid)

        fr_wheel = affinity.rotate(fr_wheel, -phi, use_radians=True)
        fl_wheel = affinity.rotate(fl_wheel, -phi, use_radians=True)
        return robot, [fr_wheel, fl_wheel]
    
    def sample_task(self):
        point1 = self.sample_valid_point()
        point2 = self.sample_valid_point()
        point1.value[2] = 0
        point2.value[2] = 0
        return self.make_state(point1.value), self.make_state(point2.value)

class PlanarMobileArm(Environment):
    def __init__(self):
        self.base_width = 2
        self.base_length = 0.1
        self.arm_lengths = [1, 1, 1]

    # def decompose_state(self, state):
    #     if isinstance(state, AngularNumpyState):
    #         return state.value
    #     elif isinstance(state, np.ndarray):
    #         return state
    #     else:
    #         raise ValueError("state input type is invalid")

    def create_end_effector_representation(self, base_point : np.ndarray):
        x, y = base_point

        y_offset = 0.5
        lines = [
            LineString([x, y], [x, y+y_offset]),
            LineString([x, y], [x, y+y_offset]),
                ]

    def generate_robot_representation(self, state):
        x, y, theta1, theta2, theta3 = state.value
        
        robot = [
            self.create_rectangle_geometry(x_loc=x, y_loc=y, x_width=self.base_width, y_length=self.base_length),
        ]
        # print([x,y+self.base_length/2], [x,y+self.base_length/2 + self.arm_lengths[0]])
        # [(x,y+self.base_length/2), (x,y+self.base_length/2 + self.arm_lengths[0])]
        # arm1 = LineString([(x,y+self.base_length/2), (x,y+self.base_length/2 + self.arm_lengths[0])])
        arm1 = LineString([(x,y+self.base_length/2), (x+self.arm_lengths[0],y+self.base_length/2)])
        arm1 = affinity.rotate(arm1, angle=theta1, use_radians=True, origin=(x,y+self.base_length/2))

        # rotated_robot = affinity.rotate(robot, theta, use_radians=True)
        # return rotated_robot

        # end_effector = 

        return robot, arm1

    def draw_environment(self, ax):
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
    def draw_state(self, ax, state):
        robot, arm = self.generate_robot_representation(state)
        ax.plot(*robot[0].exterior.xy, color='red')
        ax.plot(*arm.xy, color='red')





if __name__ == '__main__':

    env = PlanarMobileArm()
    state = AngularNumpyState(value=np.array([0, 0, np.pi/2*0.5, 0, 0.0]), angular_dims_start=2)
    # env = DubinsCarEnv()
    # state = env.sample_point()
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()
    
    # print(env.is_within_boundary(state))
    # plt.show()

    # np.random.seed(0)
    # env = CarParkingEnv()
    # state = env.make_state(np.array([6.0, 7.0, np.pi/np.sqrt(2)]))
    # controls = env.sample_controls()
    # env.draw_environment(plt.gca())
    # env.draw_state(plt.gca(), env.make_state(np.array([5,-5,45.0])))
    # plt.show()

    # print(state.value, controls.value)

    # print(env.simulate_forward(state, controls).value)


    # env.draw_environment(plt.gca())
    # env.draw_state(plt.gca(), env.make_state(np.array([5,-5,45.0])))
    # plt.show()

    # print(env.is_valid_edge(env.make_state(np.array([5, 5, -np.pi])), env.make_state(np.array([5, 5, 3.0]))))

    # print(env.is_valid_edge(env.make_state(np.array([0, 0, -np.pi])), env.make_state(np.array([1, 2, 3.0]))))
    # print(interpolate_edge(np.array([0, 0, -np.pi]), np.array([1, 2, 3.0]), delta=0.5))