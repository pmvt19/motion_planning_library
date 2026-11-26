import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from motion_planning.obstacle_sets import TestSet
from motion_planning.space import PlanarMobileArm, RobotSpace
from motion_planning.tools import NumpyState

class ApproximationSpace(RobotSpace):
    def __init__(
        self,
        space: RobotSpace,
        batch_size: int = 1000,
        do_overapproximation: bool = False,
    ):
        super().__init__()
        self.space: RobotSpace = space
        self.do_overapproximation: bool = do_overapproximation
        self.batch_size: int = batch_size
        self.obstacle_circles: np.ndarray = self.space_to_circles()

        # Set Parameters from Base Environment
        self.state_dims: int = self.space.state_dims
        self.c_space_lower_bound: np.ndarray = self.space.c_space_lower_bound
        self.c_space_upper_bound: np.ndarray = self.space.c_space_upper_bound

    def dist(self, state1: NumpyState, state2: NumpyState) -> float:
        return self.space.dist(state1, state2)
    
    def obstacles_to_aabb(self, obstacles: list) -> np.ndarray:
        aabbs = []
        for obs in obstacles:
            xs, ys = obs.exterior.xy
            x = np.min(xs)
            y = np.min(ys)
            w = np.max(xs) - x
            h = np.max(ys) - y

            aabbs.append([x+(w/2), y+(h/2), w, h])
        return np.array(aabbs)

    def space_to_circles(self) -> np.ndarray:
        aabbs = self.obstacles_to_aabb(self.space.obstacles)
        obst_circles = self.optimized_rectangle_to_circles(aabbs)
        return obst_circles
    
    def states_to_circles(self, states: np.ndarray) -> np.ndarray:
        # States : (B, d)
        representations = self.space.batch_get_robot_representations(states)
        B, *_ = states.shape
        rect_circles = self.optimized_rectangle_to_circles(
            representations['rectangles']
        ).reshape(B, -1, 3)
        seg_circles = self.segments_to_circles(
            representations['segments'], representations['segments_radii']
        ).reshape(B, -1, 3)
        point_circles = self.points_to_circles(
            representations['points'], representations['points_radius']
        ).reshape(B, -1, 3)
        state_circles = np.concatenate(
            (rect_circles, seg_circles, point_circles), axis=1
        )
        return state_circles

    def rectangles_to_circles(self, aa_rect: np.ndarray) -> np.ndarray:
        # Let's say that we get the output of this as 
        # (x,y,w,h) -> Thus would have a shape of (N,4)
        # The way I was thinking about it, one would need to loop through each
        # rectangle to create the points
        # While this isn't necessarily a problem for the environment
        # (since we can just create this once and leave it)
        # It might be slightly problematic for the robot approximation

        # self.space.obstacles
        # aa_rect = get_axis_aligned_rectangles(self.space.obstacles)
        # N, _ = aa_rect.shape

        # aa_rect = None

        # aa_rect = np.array([
        #     # [0, 0, 5, 8.0],
        #     [0, 0, 8, 5.0],
        #     # [3, 4, 3, 3],
        #     # [0, 0, 5, 5.0],
        # ])

        N = len(aa_rect)
        circles = []
        for i in range(N):
            x, y, w, h = aa_rect[i]
            r = min(w, h)/2
            # inflated_r = r * math.sqrt(2)
            if self.do_overapproximation:
                inflated_r = r * math.sqrt(2)
            else:
                inflated_r = r

            if w < h:
                length = h
                num_circles = math.ceil(length/(2*r)) #+ 1
                gap = (length-(2*r))/(num_circles-1)
                for j in range(num_circles):
                    circles.append((x, y+r+(gap*j)-(h/2), inflated_r))

            elif h < w:
                length = w
                num_circles = math.ceil(length/(2*r)) #+ 1
                gap = (length-(2*r))/(num_circles-1)

                for j in range(num_circles):
                    circles.append((x+r+(gap*j)-(w/2), y, inflated_r))
            elif h == w:
                circles.append((x,y,inflated_r))

        circles = np.array(circles)
        return circles
    
    def optimized_rectangle_to_circles(self, aa_rect: np.ndarray) -> np.ndarray:
        # aa_rect : (B, 4) -> (x, y, w, h) where x and y is the center of the rectangle
        min_dims = np.argmin(aa_rect[:, 2:], axis=1)
        vert_rect = aa_rect[min_dims == 0]
        xs = vert_rect[:, 0] # (V,)
        min_ys = vert_rect[:, 1] - vert_rect[:, 3]/2 # (V,)
        max_ys = vert_rect[:, 1] + vert_rect[:, 3]/2 # (V,)
        vert_radii = vert_rect[:, 2].reshape(-1, 1) # radii # (V,1)

        vert_starts = np.stack((xs, min_ys), axis=1) # (V,2)
        vert_ends = np.stack((xs, max_ys), axis=1) # (V,2)
        vert_segments = np.stack((vert_starts, vert_ends), axis=2).transpose(0, 2, 1) # (V,2,2)

        horiz_rect = aa_rect[min_dims == 1]
        min_xs = horiz_rect[:, 0] - horiz_rect[:, 2]/2
        max_xs = horiz_rect[:, 0] + horiz_rect[:, 2]/2
        ys = horiz_rect[:, 1]
        horiz_radii = horiz_rect[:, 3].reshape(-1, 1)
        horiz_starts = np.stack((min_xs, ys), axis=1)
        horiz_ends = np.stack((max_xs, ys), axis=1)
        horiz_segments = np.stack((horiz_starts, horiz_ends), axis=2).transpose(0, 2, 1)

        segments = np.concatenate((vert_segments, horiz_segments), axis=0)
        radii = np.concatenate((vert_radii, horiz_radii), axis=0) / 2
        circles = self.segments_to_circles(segments, radii)
        if self.do_overapproximation:
            circles[:, 2] = circles[:, 2] * math.sqrt(2)
        return circles

    def segments_to_circles(
        self, segments: np.ndarray, radius: float | np.ndarray
    ) -> np.ndarray:
        """
            B : Number of entries in the batch
            2 : This value is fixed at two since a segment must have only 2 end points
            2 : Dimension of the segment
        """
        # segments : (B, 2, 2)
        B, *_ = segments.shape
        if B == 0:
            return np.empty((0, 3))

        start_points = segments[:, 0, :] # (B, 2)
        end_points = segments[:, 1, :] # (B, 2)

        batch_rays = end_points - start_points
        segment_lengths = np.linalg.norm(batch_rays, axis=1).reshape(-1, 1) # (B,1)

        num_distinct_segment_lengths = len(np.unique(np.round(segment_lengths, 10)))

        batch_normalized_rays = batch_rays / segment_lengths # (B, 2)
        modified_segment_lengths = segment_lengths - (2*radius)

        num_circles_per_segment = np.ceil(np.round((modified_segment_lengths / (2*radius)), 10)).astype(np.int32)
        max_num_circles = math.ceil(np.max(num_circles_per_segment)) + 1

        circle_start_points = start_points + (batch_normalized_rays * radius)

        gaps = (modified_segment_lengths / num_circles_per_segment)

        batch_scaled_rays = (batch_normalized_rays * gaps.reshape(-1, 1)).reshape(-1, 1, 2)
        repeated_rays = np.repeat(batch_scaled_rays, (max_num_circles), axis=1)
        repeated_rays[:, 0, :] = 0

        trajectories = np.cumsum(repeated_rays, axis=1) + circle_start_points.reshape(-1, 1, 2)

        if isinstance(radius, float):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        elif isinstance(radius, np.ndarray):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius.reshape(-1, 1, 1)

        circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)

        if num_distinct_segment_lengths == 1:
            circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 3)
        else:
            num_circles_per_segment = num_circles_per_segment.squeeze()
            circle_center_radius_pairs = np.vstack([circle_center_radius_pairs[i, :(num_circles+1)] for i, num_circles in enumerate(num_circles_per_segment)])

        return circle_center_radius_pairs

    def optimized_segments_to_circles(
        self, segments: np.ndarray, radius: float | np.ndarray
    ) -> np.ndarray:
        """
            B : Number of entries in the batch
            2 : This value is fixed at two since a segment must have only 2 end points
            2 : Dimension of the segment
        """
        # segments : (B, 2, 2)
        B, *_ = segments.shape
        if B == 0:
            return np.empty((0, 3))

        start_points = segments[:, 0, :] # (B, 2)
        end_points = segments[:, 1, :] # (B, 2)

        batch_rays = end_points - start_points
        segment_lengths = np.linalg.norm(batch_rays, axis=1).reshape(-1, 1) # (B,1)

        num_distinct_segment_lengths = len(np.unique(np.round(segment_lengths, 10)))

        batch_normalized_rays = batch_rays / segment_lengths # (B, 2)
        modified_segment_lengths = segment_lengths - (2*radius)

        num_circles_per_segment = np.ceil(np.round((modified_segment_lengths / (2*radius)), 10)).astype(np.int32)
        max_num_circles = math.ceil(np.max(num_circles_per_segment)) + 1

        circle_start_points = start_points + (batch_normalized_rays * radius)

        gaps = (modified_segment_lengths / max_num_circles)

        batch_scaled_rays = (batch_normalized_rays * gaps.reshape(-1, 1)).reshape(-1, 1, 2)
        repeated_rays = np.repeat(batch_scaled_rays, (max_num_circles+1), axis=1)
        repeated_rays[:, 0, :] = 0

        trajectories = np.cumsum(repeated_rays, axis=1) + circle_start_points.reshape(-1, 1, 2)

        if isinstance(radius, float):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        elif isinstance(radius, np.ndarray):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius.reshape(-1, 1, 1)

        circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)
        circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 3)

        return circle_center_radius_pairs

    def points_to_circles(self, points: np.ndarray, radius: float) -> np.ndarray:
        # points: (B, 2)
        radii = np.ones((points.shape[0], 1)) * radius
        zero_radius_circles = np.concatenate((points, radii), axis=1)
        return zero_radius_circles

    def circles_to_validity(
        self, obstacle_circles: np.ndarray, robot_circles: np.ndarray
    ) -> np.ndarray:
        B = robot_circles.shape[0]
        robot_xy = robot_circles[:, :, :2]
        obst_xy = obstacle_circles[:, :2]
        distance_mat = np.sqrt(np.sum(robot_xy**2, axis=2, keepdims=True) + np.sum(obst_xy**2, axis=1, keepdims=True).T + (-2 * (robot_xy @ obst_xy.T)))

        min_dists = robot_circles[:, :, 2].reshape(B, -1, 1) + obstacle_circles[:, 2].reshape(1, 1, -1)

        validity_mask = distance_mat > min_dists
        validity_mask = validity_mask.reshape(B, -1)
        validities = np.all(validity_mask, axis=1)

        return validities
    
    def circles_to_indiv_validity(
        self, obstacle_circles: np.ndarray, robot_circles: np.ndarray
    ) -> np.ndarray:
        B = robot_circles.shape[0]
        robot_xy = robot_circles[:, :, :2]
        obst_xy = obstacle_circles[:, :2]
        distance_mat = np.sqrt(np.sum(robot_xy**2, axis=2, keepdims=True) + np.sum(obst_xy**2, axis=1, keepdims=True).T + (-2 * (robot_xy @ obst_xy.T)))

        min_dists = robot_circles[:, :, 2].reshape(B, -1, 1) + obstacle_circles[:, 2].reshape(1, 1, -1)

        validity_mask = distance_mat > min_dists
        return validity_mask
    
    def batch_is_valid(self, states: np.ndarray) -> np.ndarray:
        robot_circles = self.states_to_circles(states)
        B = robot_circles.shape[0]
        self.num_collision_checks += B
        stacked_validities = []
        num_batches = math.ceil(B / self.batch_size)
        for i in range(num_batches):
            idx_start = i * self.batch_size
            idx_end = min((i+1)*self.batch_size, B)
            validities = self.circles_to_validity(
                self.obstacle_circles, robot_circles[idx_start:idx_end]
            )
            stacked_validities.append(validities)
        stacked_validities = np.hstack(stacked_validities)
        return stacked_validities

    def draw_state(self, ax: plt.Axes, state: NumpyState) -> None:
        circles = self.states_to_circles(np.array([self.get_state_value(state)]))[0]
        patch_list = [patches.Circle((x,y), r) for (x,y,r) in circles]
        patch_collection = PatchCollection(patch_list, color='red')
        ax.add_collection(patch_collection)

    def draw_environment(self, ax: plt.Axes) -> None:
        ax.set_xlim(self.space.x_range[0], self.space.x_range[1])
        ax.set_ylim(self.space.y_range[0], self.space.y_range[1])
        circles = self.space_to_circles()

        patch_list = [patches.Circle((x,y), r) for (x,y,r) in circles]
        patch_collection = PatchCollection(patch_list, color='blue')
        ax.add_collection(patch_collection)
    
    def sample_point(self) -> NumpyState:
        return self.space.sample_point()

    def is_valid(self, state: NumpyState) -> bool:
        state = self.get_state_value(state)
        return self.batch_is_valid(state.reshape(1, -1))
    
    def make_state(self, state: np.ndarray) -> NumpyState:
        return self.space.make_state(state)

    def batch_sample_points(self, num_points: int) -> np.ndarray:
        return self.space.batch_sample_points(num_points=num_points)
    
    def batch_sample_points_around_target(self, targets: np.ndarray) -> np.ndarray:
        return self.space.batch_sample_points_around_target(targets)
    
if __name__ == "__main__":
    env = PlanarMobileArm()
    env.set_obstacles(TestSet())
    state = env.make_state(np.array([-4.0,3.0,np.pi/2,np.pi+0.2,np.pi+0.2]))
    env = ApproximationSpace(env, do_overapproximation=True)
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    env.space.draw_state(plt.gca(), state)
    plt.show()
    