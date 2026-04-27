import numpy as np

from shapely import affinity

from motion_planning.space import NonHolonomicRobot
from motion_planning.tools import NumpyState, AngularNumpyState
from motion_planning.utils import numpystate_distance, create_rectangle_geometry
from motion_planning.controller.xbox_controller import XboxController

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
    
    def clip_state(self, state : np.ndarray):
        state = np.clip(state, 
                        np.array([self.x_range[0], self.y_range[0], self.velocity_range[0], self.phi_range[0], -np.inf]), 
                        np.array([self.x_range[1], self.y_range[1], self.velocity_range[1], self.phi_range[1], np.inf]))
        return state
    
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
        theta -= np.pi/2
        state = np.array([x, y, v, phi, theta])
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
        # theta += np.pi/2 # Hack to treat the upward direction as the 0 radians orientation (Should Fix)
        x_dot = np.array([
                    v * np.cos(theta) * self.dt,
                    v * np.sin(theta) * self.dt,
                    a * self.dt,
                    psi * self.dt,
                    v / self.car_length * np.tan(phi) * self.dt,
                ])
        return x_dot
    
    def input_to_control(self, inputs):
        steer = -inputs[XboxController.XboxControls.LTHUMBX]
        accel = -inputs[XboxController.XboxControls.RTHUMBY]
        
        controls = np.array([accel * self.accel_range[1], steer * self.psi_range[1]])
        return self.make_control(np.array(controls))