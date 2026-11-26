import math
import time

import numpy as np
import rerun as rr
from sklearn.metrics import pairwise_distances

from motion_planning.obstacle_sets import ObstacleSet
from motion_planning.space import RobotSpace3D
from motion_planning.tools import AngularNumpyState, NumpyState
from motion_planning.utils import numpystate_distance, interpolate_path


class ApproximationSpace3D(RobotSpace3D):
    def __init__(self, space: RobotSpace3D, batch_size=1000, do_overapproximation=False):
        self.space = space
        self.batch_size = batch_size
        self.do_overapproximation = do_overapproximation

        self.edge_validity_delta = 0.1
        self.angular_dims_start = None

        self.obstacle_circles = self.space_to_circles()

        self.num_collision_checks = 0

        # Set Parameters from Base Environment
        self.state_dims = self.space.state_dims
        self.c_space_lower_bound = self.space.c_space_lower_bound
        self.c_space_upper_bound = self.space.c_space_upper_bound
    
    def obstacles_to_3d_aabb(self, obstacles): # TODO: Needs to be implemented properly with representations of 3d obsts
        aabbs = []
        for obs in obstacles: # These representations should be different
            aabbs.append(obs)
        return np.array(aabbs)
    
    def space_to_circles(self):
        aabbs = self.obstacles_to_3d_aabb(self.space.obstacles)
        obst_circles = self.prisms_to_circles(aabbs)
        return obst_circles

    def dist(self, state1, state2):
        return self.space.dist(state1, state2)

    def states_to_circles(self, states):
        representations = self.space.batch_get_robot_representations(states)
        B, *_ = states.shape

        prism_circles = self.prisms_to_circles(
            representations['prisms']
        ).reshape(B, -1, 4)

        cylinder_circles = self.cylinders_to_circles(
            representations['cylinders'], 
            representations['cylinder_radii']
        ).reshape(B, -1, 4)

        point_circles = self.points_to_circles(representations['points'], representations['points_radii']).reshape(B, -1, 4)
        # print(prism_circles.shape, cylinder_circles.shape, point_circles.shape)
        state_circles = np.concatenate((prism_circles, cylinder_circles, point_circles), axis=1)
        return state_circles

    def rect_prism_to_circles_x_short(self, aa_rect_prism):
        # aa_rect (x,y,z,xl,yl,zl)

        radii = aa_rect_prism[3] / 2
        x, y, z, xl, yl, zl = aa_rect_prism

        segment_lengths = yl - (2*radii)
        num_points = math.ceil(segment_lengths / (2*radii))
        delta = segment_lengths / num_points

        ys = [y-yl/2+radii]
        for i in range(num_points):
            ys.append(ys[-1] + delta)

        segment_lengths = zl - (2*radii)
        num_points = math.ceil(segment_lengths / (2*radii))
        delta = segment_lengths / num_points

        zs = [z-zl/2+radii]
        for i in range(num_points):
            zs.append(zs[-1] + delta)

        ys = np.array(ys)
        zs = np.array(zs)

        output = np.array(np.meshgrid(ys,zs)).T.reshape(-1, 2)
        points = np.hstack((np.ones((output.shape[0],1))*x, output, np.ones((output.shape[0],1))*xl/2))

        return points

    def rect_prism_to_circles_y_short(self, aa_rect_prism):
        # aa_rect (x,y,z,xl,yl,zl)

        radii = aa_rect_prism[4] / 2
        x, y, z, xl, yl, zl = aa_rect_prism

        segment_lengths = xl - (2*radii)
        num_points = math.ceil(segment_lengths / (2*radii))
        delta = segment_lengths / num_points

        xs = [x-xl/2+radii]
        for i in range(num_points):
            xs.append(xs[-1] + delta)

        segment_lengths = zl - (2*radii)
        num_points = math.ceil(segment_lengths / (2*radii))
        delta = segment_lengths / num_points

        zs = [z-zl/2+radii]
        for i in range(num_points):
            zs.append(zs[-1] + delta)

        xs = np.array(xs)
        zs = np.array(zs)

        output = np.array(np.meshgrid(xs,zs)).T.reshape(-1, 2)
        points = np.hstack((output[:, 0].reshape(-1, 1), np.ones((output.shape[0],1))*y, output[:, 1].reshape(-1, 1), np.ones((output.shape[0],1))*yl/2))

        return points

    def rect_prism_to_circles_z_short(self, aa_rect_prism):
        # aa_rect (x,y,z,xl,yl,zl)

        radii = aa_rect_prism[5] / 2
        x, y, z, xl, yl, zl = aa_rect_prism

        segment_lengths = xl - (2*radii)
        num_points = math.ceil(segment_lengths / (2*radii))
        delta = segment_lengths / num_points

        xs = [x-xl/2+radii]
        for i in range(num_points):
            xs.append(xs[-1] + delta)

        segment_lengths = yl - (2*radii)
        num_points = math.ceil(segment_lengths / (2*radii))
        delta = segment_lengths / num_points

        ys = [y-yl/2+radii]
        for i in range(num_points):
            ys.append(ys[-1] + delta)

        xs = np.array(xs)
        ys = np.array(ys)

        output = np.array(np.meshgrid(xs,ys)).T.reshape(-1, 2)
        points = np.hstack((output, np.ones((output.shape[0],1))*z, np.ones((output.shape[0],1))*zl/2))

        return points

    def rect_prism_to_circles(self, aa_rect_prism):
        # Work in progress
        # Need to handle multiple prism sizes

        # B, _ = aa_rect_prisms.shape
        
        # dim_sizes = aa_rect_prisms[:, 3:6]
        # min_dims = np.argmin(dim_sizes, axis=1)

        # radii = dim_sizes[np.arange(B), min_dims]
        pass

    def prisms_to_circles(self, aa_rect_prisms):
        """
        aa_rect_prisms: (B, 6)
        """

        if len(aa_rect_prisms) == 0:
            return np.empty((0, 4))
        
        dim_sizes = aa_rect_prisms[:, 3:6]
        min_dims = np.argmin(dim_sizes, axis=1)

        x_min_dim_mask = min_dims==0
        y_min_dim_mask = min_dims==1
        z_min_dim_mask = min_dims==2

        x_min_prisms = aa_rect_prisms[x_min_dim_mask]
        y_min_prisms = aa_rect_prisms[y_min_dim_mask]
        z_min_prisms = aa_rect_prisms[z_min_dim_mask]

        circles = []
        for prism in x_min_prisms:
            circles.append(self.rect_prism_to_circles_x_short(prism).reshape(-1, 4))
        for prism in y_min_prisms:
            circles.append(self.rect_prism_to_circles_y_short(prism).reshape(-1, 4))
        for prism in z_min_prisms:
            circles.append(self.rect_prism_to_circles_z_short(prism).reshape(-1, 4))

        return np.concatenate(circles, axis=0)

    def cylinders_to_circles(self, cylinders, radius):
        """
        cylinders: (Bm, m, 2, 3)
        cyl_radii: (Bm, 1) or scaler

        returns: (Bm, m, 4)
        """
        Bm, m, _, _ = cylinders.shape

        if len(cylinders) == 0:
            return np.empty((0, m, 4))
        
        cylinders = cylinders.reshape(Bm * m, 2, 3)

        start_points = cylinders[:, 0, :] # (B, 3)
        end_points = cylinders[:, 1, :] # (B, 3)

        batch_rays = end_points - start_points
        segment_lengths = np.linalg.norm(batch_rays, axis=1).reshape(-1, 1) # (B,1)

        num_distinct_segment_lengths = len(np.unique(np.round(segment_lengths, 10)))
    
        batch_normalized_rays = batch_rays / segment_lengths # (B, 3)
        modified_segment_lengths = segment_lengths - (2*radius)

        num_circles_per_segment = np.ceil(np.round((modified_segment_lengths / (2*radius)), 10)).astype(np.int32)
        max_num_circles = math.ceil(np.max(num_circles_per_segment)) + 1

        circle_start_points = start_points + (batch_normalized_rays * radius)

        gaps = (modified_segment_lengths / num_circles_per_segment)

        batch_scaled_rays = (batch_normalized_rays * gaps.reshape(-1, 1)).reshape(-1, 1, 3)
        repeated_rays = np.repeat(batch_scaled_rays, (max_num_circles), axis=1)
        repeated_rays[:, 0, :] = 0

        trajectories = np.cumsum(repeated_rays, axis=1) + circle_start_points.reshape(-1, 1, 3)

        if isinstance(radius, float):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        elif isinstance(radius, np.ndarray):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius.reshape(-1, 1, 1)

        circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)

        if num_distinct_segment_lengths == 1:
            circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 4)
        else:
            num_circles_per_segment = num_circles_per_segment.squeeze()
            circle_center_radius_pairs = np.vstack([circle_center_radius_pairs[i, :(num_circles+1)] for i, num_circles in enumerate(num_circles_per_segment)])

        circle_center_radius_pairs = circle_center_radius_pairs.reshape(Bm, -1, 4)
        return circle_center_radius_pairs

    def points_to_circles(self, points, radii):
        shaped_radii = np.ones((points.shape[0], 1)) * radii
        radius_circles = np.concatenate((points, shaped_radii), axis=1)
        return radius_circles

    def circles_to_validity(self, obstacle_circles, robot_circles):
        """
        self.
        robot_circles: (B, N, 4)
        """
        B = robot_circles.shape[0]

        obst_xyz = obstacle_circles[:, :3]
        robot_xyz = robot_circles[:, :, :3]

        distance_mat = np.sqrt(np.sum(robot_xyz**2, axis=2, keepdims=True) + np.sum(obst_xyz**2, axis=1, keepdims=True).T + (-2 * (robot_xyz @ obst_xyz.T)))

        min_dists = robot_circles[:, :, 3].reshape(B, -1, 1) + obstacle_circles[:, 3].reshape(1, 1, -1)

        validity_mask = distance_mat > min_dists
        validity_mask = validity_mask.reshape(B, -1)
        validities = np.all(validity_mask, axis=1)
        return validities

    def batch_is_valid(self, states):
        robot_circles = self.states_to_circles(states)
        B = robot_circles.shape[0]
        self.num_collision_checks += B
        stacked_validities = []
        num_batches = math.ceil(B / self.batch_size)
        for i in range(num_batches):
            idx_start = i * self.batch_size
            idx_end = min((i+1)*self.batch_size, B)
            validities = self.circles_to_validity(self.obstacle_circles, robot_circles[idx_start:idx_end])
            stacked_validities.append(validities)
        stacked_validities = np.hstack(stacked_validities)
        return stacked_validities

    def draw_state(self, ax, state, method='rerun'):
        rr.init("3D Environment", spawn=True)
        state_circles = self.states_to_circles(np.array(state.value).reshape(1, -1))[0]
        rr.log("State Spheres", rr.Points3D([state_circles[:, :3]], colors=[255, 0, 0], radii=state_circles[:, 3]))

    def draw_environment(self, ax, method='rerun'):
        rr.init("3D Environment", spawn=True)
        rr.log("Obstacle Spheres", rr.Points3D([self.obstacle_circles[:, :3]], colors=[0, 0, 255], radii=self.obstacle_circles[:, 3]))

    def sample_point(self):
        return self.space.sample_point()
    
    def is_valid(self, state):
        state = self.get_state_value(state)
        return self.batch_is_valid(state.reshape(1, -1))

    def make_state(self, state):
        return self.space.make_state(state)

    def batch_sample_points_around_target(self, targets):
        raise NotImplementedError

if __name__ == '__main__':

    center = (0,0,0)
    lengths = (2,2,2)

    edges = np.array([
        [0,1], # Bottom Face
        [1,2],
        [2,3],
        [3,0], # Bottom Face
        
        [4,5], # Top Face
        [5,6],
        [6,7],
        [7,4], # Top Face

        [0,4], # Mid Faces
        [1,5],
        [2,6],
        [3,7], # Mid Faces
    ])

    from motion_planning.utils import set_numpy_seed
    from motion_planning.space.holonomic._3d.ur5 import UR5
    from motion_planning.obstacle_sets import SingleCup
    # set_numpy_seed(5568)
    env_base = UR5()
    env_base.set_obstacles(SingleCup())
    env = ApproximationSpace3D(env_base)
    # state = env.make_state(np.array([0.0, -np.pi/4, np.pi/5, 0.0]))
    state = env.make_state(np.array([np.pi/4, -np.pi/4, np.pi/4, np.pi/4]))

    # env.draw_environment(None, None, method='rerun')
    # state = env.make_state(np.array([0.0, -np.pi/4, np.pi/2, np.pi/2]))
    # state = env.sample_point()
    state = env.make_state(np.array([0.0, -np.pi/4, np.pi/5, 0.0]))

    from motion_planning.search import PRM, RRT
    # prm = PRM(env, num_samples=1000, num_neighbors=5, validate_edges=True)
    prm = PRM(env, num_samples=10000, num_neighbors=10, validate_edges=True)
    # prm = PRM(env, num_samples=100000, edge_dist_radius=0.5, validate_edges=True)
    # rrt = RRT(env, delta=1.0)

    target_points = env_base.sample_configs_ee_target(np.array([2.5, 0, 1.0]))
    print(target_points.shape)

    st = time.time()
    prm.create_graph(starting_samples=target_points)
    et = time.time()
    print(et-st, "Time to create prm graph")
    start, target = env.sample_point(), env.sample_point()
    target = state
    start = env.make_state(np.array([0.0, 0.0, 0.0, 0.0]))

    path = prm.search(start, target)
    before_path = path
    # path = rrt.search(start, target, max_steps=100000)
    path = interpolate_path(path, env, 0.1)
    env.animate_path(path, method='rerun')

    for state in path.path:
        print(env.is_valid(state))

    numpy_path = np.array([state.value for state in path.path])
    before_numpy_path = np.array([state.value for state in before_path.path])
    
    path_edges = np.concatenate((numpy_path[:-1], numpy_path[1:]), axis=1)
