import math
import time

import numpy as np
import rerun as rr
from sklearn.metrics import pairwise_distances

from motion_planning.obstacle_sets import ObstacleSet
from motion_planning.space import HolonomicRobot, RobotSpace
from motion_planning.tools import AngularNumpyState, NumpyState
from motion_planning.utils import numpystate_distance, interpolate_path


class SphereRobot(HolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.1

        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.z_range = [-10,10]

        # self.theta_range = [0, 2*np.pi]
        # self.angular_dims_start = 2

        self.robot_radius = 0.5

        self.obstacles = []

        self.do_boundary_check = True

        ### HARD CODED ###
        aa_rect_prism1 = np.array([0,0,0,1,5,5])
        aa_rect_prism2 = np.array([2.5,2.5,0,5,1,5])
        aa_rect_prism3 = np.array([2.5,-2.5,0,5,1,5])
        aa_rect_prism4 = np.array([2.5,0,-2.5,5,5,1])
        aa_rect_prism5 = np.array([2.5,0,2.5,5,5,1])

        # prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3])
        prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3, aa_rect_prism4, aa_rect_prism5])
        self.obstacles = prisms
        ### HARD CODED ###

        # TODO: Add CSpace Dimensions
        # TODO: Fix Obstacle Setting

        self.num_collision_checks = 0
    
    def make_state(self, state):
        return NumpyState(value=state)

    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        z = np.random.uniform(low=self.z_range[0], high=self.z_range[1])
        return self.make_state(np.array([x, y, z]))
    
    def generate_robot_representation(self, state, r):
        xCenter, yCenter, zCenter = self.get_state_value(state)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)
        # shift and scale sphere
        x = r*x + xCenter
        y = r*y + yCenter
        z = r*z + zCenter
        return (x,y,z)

    def dist(self, state1, state2):
        return numpystate_distance(state1, state2)

    def is_valid(self, state):
        raise NotImplementedError
    
    def draw_state(self, ax, state):
        cpx, cpy, cpz = self.generate_robot_representation(state)
        # ax.plot_surface(cpx, cpy, cpz, color="r")
        ax.plot_wireframe(cpx, cpy, cpz, color="r")

    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.set_zlim(self.z_range[0], self.z_range[1])
        
        for prism in self.obstacles:
            ordered_verts = xyzwhl_to_ordered_vertices(prism)
            for a,b in edges:
                point_a = ordered_verts[a]
                point_b = ordered_verts[b]
                ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]], color='blue')

    def batch_get_robot_representations(self, states):
        return {
            'prisms' : np.empty((0, 6)),
            'cylinders' : np.empty((0, 0, 2, 3)), 
            'cylinder_radii' : 0.0,
            'points' : states, 
            'points_radii' : self.robot_radius
        }
    
    def batch_sample_points_around_target(self, targets):
        validities = self.batch_is_valid(targets)
        return targets[validities]