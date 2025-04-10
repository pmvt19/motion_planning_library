import numpy as np
from shapely import Polygon, Point, LineString, affinity
from state import NumpyState, AngularNumpyState
import matplotlib.pyplot as plt
from obstacle_sets import ObstacleSet
from utils import create_rectangle_geometry, numpystate_distance, interpolate_SE2_edge, issue_warning, interpolate_edge
import time
from collections import defaultdict

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
        
    def get_edge_states(self, start, end):
        raise NotImplementedError
        
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
    
    def set_obstacles(self, obstacle_set : ObstacleSet):
        self.obstacles = obstacle_set.obstacles
        self.boundary = obstacle_set.boundary

        # print(self.boundary.exterior.xy)
        x_points, y_points = self.boundary.exterior.xy
        self.x_range = [min(x_points), max(x_points)]
        self.y_range = [min(y_points), max(y_points)]
    
    def batch_get_edge_states(self, start_states, end_states):
        raise NotImplementedError
    
    def batch_is_valid(self, states : np.ndarray):
        validities = []
        for state in states:
            validities.append(self.is_valid(self.make_state(state)))
        validities = np.array(validities)
        return validities
    
    def batch_is_valid_edge(self, start_states : np.ndarray, end_states : np.ndarray):
        B, d = start_states.shape
        # start_states: (B, d), end_states: (B, d)
        time0 = time.time()
        print("WARNING: THIS IS A HACK THAT NEEDS TO BE FIXED ASAP")
        pts, steps = self.space.batch_get_edge_states(start_states, end_states) # points -> (B, max_steps, d), steps -> (B,)
        time1 = time.time()
        print(f"Time to interpolate edges: {time1-time0}")
        pts = pts.reshape(-1, d)
        time2 = time.time()
        print(pts.shape)
        pt_validities = self.batch_is_valid(pts).reshape(B, -1)
        time3 = time.time()
        print(f"Time to get state validities for size {B}: {time3-time2}")
        edge_validities = np.array([np.all(pt_validities[i, :steps[i]]) for i in range(len(steps))])
        time4 = time.time()
        print(f"Time to get edge validities from state validities: {time4-time3}")

        return edge_validities
        
    

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
        self.robot_length = 5

        self.obstacles = [
            create_rectangle_geometry(0, 0, 5, 5)
        ]

        self.do_boundary_check = True

    def make_state(self, state : np.ndarray):
        return AngularNumpyState(value=state, angular_dims_start=self.angular_dims_start)

    def generate_robot_representation(self, state):
        x, y, theta = self.get_state_value(state)
        theta -= np.pi
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

    def draw_state(self, ax, state):
        robot = self.generate_robot_representation(state)
        ax.plot(*robot.exterior.xy, color='red')
    
class PlanarMobileArm(HolonomicRobot):
    def __init__(self, num_links=3, arm_lengths=None):
        super().__init__()
        self.base_width = 2
        self.base_length = 0.1

        self.x_range = [-10, 10]
        self.y_range = [-10, 10]
        self.theta1_range = [0, np.pi]
        self.theta_range = [0, 2*np.pi]

        self.edge_validity_delta = 0.5
        # self.angular_dims_start = 2

        assert (num_links > 0), "Num Links Must Be Greater Than Zero"
        
        self.num_links = num_links
        self.arm_lengths = np.array([1] * self.num_links)

        if arm_lengths is not None:
            assert (num_links == len(arm_lengths)), "Num Links must be equal to the list size of arm lengths"
            self.arm_lengths = np.array(arm_lengths)
        
        self.obstacle_check_time = 0
        self.generate_state_time = 0

        self.timing_dict = defaultdict(float)

    def dist(self, state1 : NumpyState, state2 : NumpyState):
        state1 = self.get_state_value(state1)
        state2 = self.get_state_value(state2)
        return np.linalg.norm(state1 - state2)
    
    def make_state(self, state : np.ndarray):
        return NumpyState(value=state)
    
    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        theta1 = np.random.uniform(low=self.theta1_range[0], high=self.theta1_range[1])
        link_thetas = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1], size=(self.num_links-1,))
        return self.make_state(np.array([x, y, theta1, *link_thetas]))
    
    def create_end_effector_representation(self, base_point : np.ndarray):
        x, y = base_point

        ee_lengths = [0.5, 0.2]
        ee_angles = [np.pi/6, 2*np.pi/3]
        # cumulative_ee_angles = np.cumsum(ee_angles)

        r_joint_pos1 = np.array([
            ee_lengths[0] * np.cos(ee_angles[0]),
            ee_lengths[0] * np.sin(ee_angles[0]),
        ]) + base_point

        r_joint_pos2 = r_joint_pos1 + np.array([
            ee_lengths[1] * np.cos(ee_angles[0] + ee_angles[1]),
            ee_lengths[1] * np.sin(ee_angles[0] + ee_angles[1]),
        ])

        p1 = np.pi - ee_angles[0]
        p2 = 2*np.pi - ee_angles[1]

        l_joint_pos1 = np.array([
            ee_lengths[0] * np.cos(p1),
            ee_lengths[0] * np.sin(p1),
        ]) + base_point

        l_joint_pos2 = l_joint_pos1 + np.array([
            ee_lengths[1] * np.cos(p1+p2),
            ee_lengths[1] * np.sin(p1+p2),
        ])

        ee_point_pairs = [(base_point, r_joint_pos1),
                          (r_joint_pos1, r_joint_pos2),
                          (base_point, l_joint_pos1),
                          (l_joint_pos1, l_joint_pos2)]
        
        return ee_point_pairs
    
    def get_arm_base_position(self, state):
        state = self.get_state_value(state)
        x, y, *_ = state
        return np.array([x, y+self.base_length/2])

    def forward_kinematics(self, state):
        state = self.get_state_value(state)
        x, y, *thetas = state

        thetas = np.array(thetas)
        thetas[1:] = thetas[1:] - np.pi # Hack to treat angles properly

        arm_base = self.get_arm_base_position(state)
        link_thetas = np.cumsum(thetas)

        cos = np.cos(link_thetas)
        sin = np.sin(link_thetas)
        normalized_link_points = np.vstack((cos, sin)).T

        point_der = self.arm_lengths.reshape(self.num_links, 1) * normalized_link_points
        joint_pos = np.cumsum(point_der, axis=0) + arm_base
        return np.vstack((arm_base, joint_pos))

    def generate_robot_representation(self, state):
        time0 = time.time()
        state = self.get_state_value(state)
        x, y, *thetas = state
        time1 = time.time()
        robot = create_rectangle_geometry(x_loc=x, y_loc=y, x_width=self.base_width, y_length=self.base_length)
        time2 = time.time()
        rotation_offset = np.pi/2 if self.num_links % 2 == 0 else -np.pi/2
        arms = []

        time3 = time.time()
        joint_positions = self.forward_kinematics(state)
        time4 = time.time()
        for i in range(len(joint_positions) - 1):
            arm = LineString([joint_positions[i], joint_positions[i+1]])
            arms.append(arm)
        time5 = time.time()
        end_effector_point_pairs = self.create_end_effector_representation(joint_positions[-1])
        # end_effector_point_pairs = []
        time6 = time.time()
        ee = []
        for i in range(len(end_effector_point_pairs)):
            ee_link = LineString(end_effector_point_pairs[i])
            ee_link = affinity.rotate(ee_link, angle=(np.sum(thetas)+rotation_offset), use_radians=True, origin=list(joint_positions[-1]))
            ee.append(ee_link)
        time7 = time.time()

        self.timing_dict['get_state_value'] += time1-time0
        self.timing_dict['create_robot_base'] += time2-time1
        self.timing_dict['forward_kinematics'] += time4-time3
        self.timing_dict['create_arm_link'] += time5-time4
        self.timing_dict['create_end_effector_representation'] += time6-time5
        self.timing_dict['create_end_effector_lines'] += time7-time6

        return robot, arms, ee 

    def draw_state(self, ax, state):
        robot, arms, ee = self.generate_robot_representation(state)
        ax.plot(*robot.exterior.xy, color='red')

        for arm in arms:
            ax.plot(*arm.xy, color='red')
        
        for ee_link in ee:
            ax.plot(*ee_link.xy, color='red')

    def collides_with_self(self, robot, arms, ee):
        for i in range(len(arms)):
            for j in range(i+2, len(arms)):
                if arms[i].intersects(arms[j]):
                    return True

            if i > 0:
                if robot.intersects(arms[i]):
                    return True 
        return False 
    
    def is_valid(self, state):
        start_time = time.time()
        self.num_collision_checks += 1
        robot, arms, ee = self.generate_robot_representation(state)
        self.generate_state_time += (time.time() - start_time)
        # if self.collides_with_self(robot, arms, ee):
        #     return False
        start_time = time.time()
        for obs in self.obstacles:
            if obs.intersects(robot):
                return False
            for arm in arms:
                if obs.intersects(arm):
                    return False
            for ee_link in ee:
                if obs.intersects(ee_link):
                    return False
        self.obstacle_check_time += (time.time() - start_time)

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

    ## ---- Batched Methods ---- ##
    def batch_forward_kinematics(self, states : np.ndarray):
        states[:, 3:] -= np.pi # Hack to treat angles properly
        arm_bases = np.vstack((states[:, 0], states[:, 1] + self.base_length/2)).T # (B, 2)
        link_thetas = np.cumsum(states[:, 2:], axis=1) # (B, num_links)
        link_cosines = np.cos(link_thetas) # (B, num_links)
        link_sines = np.sin(link_thetas) # (B, num_links)
        normalized_link_points = np.stack((link_cosines, link_sines), axis=2) # (B, num_links, 2)
        point_der = self.arm_lengths.reshape(1, self.num_links, 1) * normalized_link_points # (B, num_links, 2)
        joint_pos = np.cumsum(point_der, axis=1) + arm_bases.reshape(-1, 1, 2) # (B, num_links, 2)
        return np.concatenate((arm_bases.reshape(-1, 1, 2), joint_pos), axis=1)
    
    def batch_get_robot_representations(self, states : np.ndarray):
        rectangles = np.stack((states[:, 0], states[:, 1], np.ones((states.shape[0])) * self.base_width, np.ones((states.shape[0])) * self.base_length), axis=1)
        segment_points = self.batch_forward_kinematics(states)#.reshape(-1, 2, 2)
        start_points = segment_points[:, :-1, :]
        end_points = segment_points[:, 1:, :]
        segments = np.concatenate((start_points, end_points), axis=2).reshape(-1, 4).reshape(-1, 2, 2)
        return {
            'rectangles' : rectangles,
            'segments' : segments, 
            'points' : np.empty((0, 2)),
        }
    
    def batch_get_edge_states(self, start_states : np.ndarray, end_states : np.ndarray):
        """
        This function will be generalized to utils as it is not specific to this robot space
        """
        # (B, d), # (B, d)
        B, d  = start_states.shape
        print(end_states.shape, start_states.shape)
        vecs = (end_states - start_states)
        print(vecs.shape)
        lengths = np.linalg.norm(vecs, axis=1)
        print(lengths)
        normalized_vecs = vecs / lengths.reshape(-1, 1) # TODO: Probably need to reshape # (B, d)
        
        # num_checks = int(edge_length / self.edge_validity_delta)

        num_steps = np.ceil((lengths / self.edge_validity_delta) + 1).astype(np.int32)
        print(num_steps, 'steps')
        max_steps = np.max(num_steps).astype(np.int32)
        print(max_steps, 'max steps')

        # WARNING: TILE MAY NOT WORK
        # edge_states_derivative = np.tile(normalized_vecs * self.edge_validity_delta, (max_steps)) # (B, max_steps, d)
        normalized_vecs = normalized_vecs.reshape(-1, 1, d) # Reshape normalized vectors for repeating function
        edge_states_derivative = np.repeat(normalized_vecs * self.edge_validity_delta, (max_steps), axis=1) # (B, max_steps, d)
        edge_states_derivative[:,0,:] = 0
        edge_states = np.cumsum(edge_states_derivative, axis=1) + start_states.reshape(-1, 1, d) # (B, max_steps, d) + (B,1,d)
        edge_states[np.arange(B), (num_steps-1), :] = end_states
        # padded_num_steps = np.hstack((0, num_steps)) # Padding to make it easier to index the results later
        return edge_states, num_steps

    # def batch_is_valid(self, states: np.ndarray):
    #     raise NotImplementedError
    
    # def batch_is_valid_edge(self, start_states : np.ndarray, end_states : np.ndarray):
    #     raise NotImplementedError
    
class NonHolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()

        self.dt = 0.05
    
    def make_control(self, state : np.ndarray):
        raise NotImplementedError
    def state_derivative(self, state, control):
        raise NotImplementedError

    # TODO: Implement these functions here
    def simulate(self, starting_state: NumpyState, control_seq: list):
        state = starting_state
        state_seqs = [state]
        for control, time in control_seq:
            state, _, _ = self.extend_state(state, time, control, do_collision_checking=False)
            state_seqs.append(state)
        return state_seqs
    
    def extend_state(self, state: NumpyState, time: float, controls=None, do_collision_checking=True):
        if controls is None:
            controls = self.sample_controls()

        list_of_states = [state]
        running_time = 0
        num_iterations = int(time / self.dt)

        for i in range(num_iterations):
            state = self.simulate_step(state, controls)
            if do_collision_checking and not self.is_valid(state):
                break
            running_time = (i+1) * self.dt
            list_of_states.append(state)

        return list_of_states[-1], controls, running_time
    
    def simulate_step(self, state, control):
        state = self.get_state_value(state)
        x_dot = self.state_derivative(state, control)
        return self.make_state(state + x_dot)

class SkidSteerCar(NonHolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.05

        self.car_width = 1
        self.car_length = 2

        self.angular_dim_start = 2

        self.x_range = [-10, 10]
        self.y_range = [-10, 10]
        self.theta_range = [0, 2*np.pi]

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
        v = np.random.uniform(low=self.velocity_range[0], high=self.velocity_range[1]) # Sample Velocity Uniformly Between: (-3, 3)
        delta = np.random.uniform(low=self.delta_range[0], high=self.delta_range[1]) # Sample Delta Uniformly Between: (-pi, pi)

        # Can only Move Forward OR Turn in place Not Both!
        if np.random.random() < bias:
            delta = 0.0
        else:
            v = 0.0
        return self.make_control(np.array([v, delta]))

    def make_state(self, state : np.ndarray):
        return AngularNumpyState(state, angular_dims_start=self.angular_dim_start)

    def make_control(self, control : np.ndarray):
        return NumpyState(control)
    
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
        theta -= np.pi/2
        robot = create_rectangle_geometry(x_loc=x, y_loc=y, x_width=self.car_width, y_length=self.car_length)
        robot = affinity.rotate(robot, theta, use_radians=True)
        return robot

    def generate_costmetic_robot_representation(self, state):
        print("Cosmetic State is Incomplete")
        return self.generate_robot_representation(state), []

    def draw_state(self, ax, state):
        robot, cosmetics = self.generate_costmetic_robot_representation(state)
        x,y = robot.exterior.xy
        ax.plot(x,y, color='red')

        for c in cosmetics:
            x,y = c.exterior.xy
            ax.plot(x,y, color='black')
    
    def state_derivative(self, state, control):
        x, y, theta = self.get_state_value(state)
        v, delta = self.get_state_value(control)
        x_dot = np.array([
                    v * np.cos(theta) * self.dt,
                    v * np.sin(theta) * self.dt,
                    delta * self.dt,
                ])
        return x_dot
    
    def get_edge_states(self, start, end):
        edge_states = interpolate_SE2_edge(start, end, self.edge_validity_delta)
        edge_states[:, 2] = edge_states[:, 2] % (2*np.pi)
        return edge_states


class DubinsCar(NonHolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.05

        self.x_range = [-10, 10]
        self.y_range = [-10, 10]
        self.velocity_range = [-3, 3]
        self.phi_range = [-np.pi/3, np.pi/3]
        self.theta_range = [0, 2*np.pi]

        self.accel_range = [-5, 5]
        self.psi_range = [-1, 1]

        self.boundary = create_rectangle_geometry(0, 0, 20, 20)

        self.angular_dims_start = 4

        self.car_width = 2
        self.car_length = 4

        self.dt = 0.05

        self.state_dim = 5
        self.control_dim = 2

    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        v = np.random.uniform(low=self.velocity_range[0], high=self.velocity_range[1])
        phi = np.random.uniform(low=self.phi_range[0], high=self.phi_range[1])
        theta = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        return self.make_state(np.array([x, y, v, phi, theta]))
    
    def sample_controls(self):
        a = np.random.uniform(low=self.accel_range[0], high=self.accel_range[1])
        psi = np.random.uniform(low=self.psi_range[0], high=self.psi_range[1])
        return self.make_control(np.array([a, psi]))
    
    def make_state(self, state):
        return AngularNumpyState(state, angular_dims_start=self.angular_dims_start)
    
    def make_control(self, control):
        return NumpyState(control)
    
    def dist(self, state1, state2):
        state1 = self.get_state_value(state1)
        state2 = self.get_state_value(state2)
        return numpystate_distance(self.make_state(state1), self.make_state(state2))
    
    def is_valid_state_constraints(self, state):
        x, y, v, phi, theta = self.get_state_value(state)
        if v < self.velocity_range[0] or v > self.velocity_range[1]:
            return False
        if phi < self.phi_range[0] or phi > self.phi_range[1]:
            return False
        return True

    def is_within_boundary(self, state):
        robot = self.generate_robot_representation(state)
        return robot.within(self.boundary)
    
    def is_valid(self, state):
        self.num_collision_checks += 1
        if self.is_valid_state_constraints(state) and self.is_within_boundary(state):
            robot = self.generate_robot_representation(state)
            for obs in self.obstacles:
                if obs.intersects(robot):
                    return False
            return True
        else:
            return False
    
    def generate_robot_representation(self, state):
        x, y, v, phi, theta = self.get_state_value(state)
        robot = create_rectangle_geometry(x_loc=x, y_loc=y, x_width=self.car_width, y_length=self.car_length)
        robot = affinity.rotate(robot, theta, use_radians=True)
        return robot
    
    def generate_costmetic_robot_representation(self, state):
        x, y, v, phi, theta = self.get_state_value(state)
        
        robot = self.generate_robot_representation(state)
        robot_centroid = robot.centroid

        self.x_offset = 0.1
        self.y_offset = 0.3
        self.wheel_length = 0.4
        self.wheel_width = 0.1

        fr_wheel = create_rectangle_geometry(x_loc=x+self.car_width/2+self.x_offset,
                                             y_loc=y+self.car_length/2-self.y_offset,
                                             x_width=self.wheel_width,
                                             y_length=self.wheel_length)
        
        fl_wheel = create_rectangle_geometry(x_loc=x-self.car_width/2-self.x_offset,
                                             y_loc=y+self.car_length/2-self.y_offset,
                                             x_width=self.wheel_width,
                                             y_length=self.wheel_length)
        
        fr_wheel = affinity.rotate(fr_wheel, theta, use_radians=True, origin=robot_centroid)
        fl_wheel = affinity.rotate(fl_wheel, theta, use_radians=True, origin=robot_centroid)

        fr_wheel = affinity.rotate(fr_wheel, phi, use_radians=True)
        fl_wheel = affinity.rotate(fl_wheel, phi, use_radians=True)

        max_v = np.max(np.abs(self.velocity_range))
        arrow_length = self.car_length / 2 * (v / max_v)
        velocity_arrow_stem = LineString([(x,y), (x,y+arrow_length)])
        velocity_arrow_stem = affinity.rotate(velocity_arrow_stem, theta, use_radians=True, origin=robot_centroid)

        return robot, [fr_wheel, fl_wheel, velocity_arrow_stem]
    
    def draw_state(self, ax, state):
        robot, cosmetics = self.generate_costmetic_robot_representation(state)
        x,y = robot.exterior.xy
        ax.plot(x,y, color='red')

        for c in cosmetics[:2]:
            x,y = c.exterior.xy
            ax.plot(x,y, color='black')
        ax.plot(*cosmetics[2].xy, color='blue')
    
    def state_derivative(self, state, control):
        x, y, v, phi, theta = self.get_state_value(state)
        a, psi = self.get_state_value(control)
        theta += np.pi/2 # Hack to treat the upward direction as the 0 radians orientation (Should Fix)
        x_dot = np.array([
                    v * np.cos(theta) * self.dt,
                    v * np.sin(theta) * self.dt,
                    a * self.dt,
                    psi * self.dt,
                    v / self.car_length * np.tan(phi) * self.dt,
                ])
        return x_dot
    
    def get_edge_states(self, start, end):
        start = self.get_state_value(start)
        end = self.get_state_value(end)
        # return numpystate_distance(self.make_state(start), self.make_state(end))

if __name__ == '__main__':
    np.random.seed(0)
    env = PlanarMobileArm(num_links=3, arm_lengths=[1,1])
    # env = SkidSteerCar()
    # state = env.make_state(np.array([0.0, 0.0, np.pi/2, 0, 0, 0]))
    # state = env.make_state(0)
    # state = env.sample_point()
    # env2 = GeneralizedPlanarMobileArm(num_links=3)
    # print(state.value)
    # env.draw_environment(plt.gca())
    # env.draw_state(plt.gca(), state)
    # plt.show()

    state1 = env.sample_point()
    state2 = env.sample_point()

    # env.draw_environment(plt.gca())
    # env.draw_state(plt.gca(), state1)
    # env.draw_state(plt.gca(), state2)
    # plt.show()

    states = np.array([state1.value, state2.value])
    # print(env.batch_forward_kinematics(states))


    states = [env.sample_point() for _ in range(100000)]
    start_time = time.time()
    for state in states:
        env.forward_kinematics(state)
    end_time = time.time()
    print("Unbatched Forward Kinematics Time:", end_time-start_time)

    start_time = time.time()
    states = np.array([state.value for state in states])
    env.batch_forward_kinematics(states)
    end_time = time.time()

    print("Batched Forward Kinematics Time:", end_time-start_time)

    # start, target = env.sample_task()
    # print(start.value, target.value)
    # env.get_edge_states(start.value, target.value)

    # pointrobot = PointRobot()
    # print(pointrobot.get_edge_states(start.value, target.value))
    # env.draw_environment(plt.gca())
    # env.draw_state(plt.gca(), start)
    # env.draw_state(plt.gca(), target)
    # plt.show()

