from space import RobotSpace, PlanarMobileArm, PolygonalRobot
from obstacle_sets import TestSet, NonRegularPolygonObst
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from shapely import Point, Polygon
from sklearn.metrics import pairwise_distances
import time
import torch

class ApproximationSpaceTorch(RobotSpace):
    def __init__(self, space : RobotSpace, batch_size=1000, do_overapproximation=False, device='cpu'):
        super().__init__()
        print("Warning: The Torch Version of Approximation is still in testing")

        self.space = space
        self.do_overapproximation = do_overapproximation
        self.batch_size = batch_size
        self.obstacle_circles = self.space_to_circles().type(torch.float64)
        self.device = device

        
        
    def dist(self, state1, state2):
        return self.space.dist(state1, state2)
    
    def obstacles_to_aabb(self, obstacles):
        aabbs = []
        for obs in obstacles:
            xs, ys = obs.exterior.xy

            xs = torch.tensor(xs)
            ys = torch.tensor(ys)

            x = torch.min(xs)
            y = torch.min(ys)
            w = torch.max(xs) - x
            h = torch.max(ys) - y

            aabbs.append([x+(w/2), y+(h/2), w, h])
        return torch.tensor(aabbs)

    def space_to_circles(self):
        aabbs = self.obstacles_to_aabb(self.space.obstacles)
        obst_circles = self.optimized_rectangle_to_circles(aabbs)
        return obst_circles
    
    def states_to_circles(self, states):
        # States : (B, d)
        representations = self.space.batch_get_robot_representations(states)
        B, *_ = states.shape

        rect_circles = self.optimized_rectangle_to_circles(torch.from_numpy(representations['rectangles']).to(self.device)).reshape(B, -1, 3)
        seg_circles = self.segments_to_circles(torch.from_numpy(representations['segments']).to(self.device), torch.from_numpy(representations['segments_radii']).to(self.device)).view(B, -1, 3)
        point_circles = self.points_to_circles(torch.from_numpy(representations['points']).to(self.device)).view(B, -1, 3)
        state_circles = torch.cat((rect_circles, seg_circles, point_circles), dim=1)
        print(f"State Circles Device: {state_circles.device}")
        return state_circles
    
    def optimized_rectangle_to_circles(self, aa_rect):
        # aa_rect : (B, 4) -> (x, y, w, h) where x and y is the center of the rectangle
        min_dims = torch.argmin(aa_rect[:, 2:], dim=1)
        vert_rect = aa_rect[min_dims == 0]
        xs = vert_rect[:, 0] # (V,)
        min_ys = vert_rect[:, 1] - vert_rect[:, 3]/2 # (V,)
        max_ys = vert_rect[:, 1] + vert_rect[:, 3]/2 # (V,)
        vert_radii = vert_rect[:, 2].view(-1, 1) # radii # (V,1)

        vert_starts = torch.stack((xs, min_ys), dim=1) # (V,2)
        vert_ends = torch.stack((xs, max_ys), dim=1) # (V,2)
        vert_segments = torch.stack((vert_starts, vert_ends), dim=2).permute(0, 2, 1) # (V,2,2)

        horiz_rect = aa_rect[min_dims == 1]
        min_xs = horiz_rect[:, 0] - horiz_rect[:, 2]/2
        max_xs = horiz_rect[:, 0] + horiz_rect[:, 2]/2
        ys = horiz_rect[:, 1]
        horiz_radii = horiz_rect[:, 3].reshape(-1, 1)
        horiz_starts = torch.stack((min_xs, ys), dim=1)
        horiz_ends = torch.stack((max_xs, ys), dim=1)
        horiz_segments = torch.stack((horiz_starts, horiz_ends), dim=2).permute(0, 2, 1)

        segments = torch.cat((vert_segments, horiz_segments), dim=0)
        radii = torch.cat((vert_radii, horiz_radii), dim=0) / 2
        circles = self.segments_to_circles(segments, radii)
        if self.do_overapproximation:
            circles[:, 2] = circles[:, 2] * math.sqrt(2)

        return circles

    def segments_to_circles(self, segments : torch.Tensor, radius):
        """
            B : Number of entries in the batch
            2 : This value is fixed at two since a segment must have only 2 end points
            2 : Dimension of the segment
        """

        # segments : (B, 2, 2)
        B, *_ = segments.shape
        if B == 0:
            return torch.empty((0, 3))

        start_points = segments[:, 0, :] # (B, 2)
        end_points = segments[:, 1, :] # (B, 2)

        batch_rays = end_points - start_points
        segment_lengths = torch.linalg.norm(batch_rays, dim=1).view(-1, 1) # (B,1)

        num_distinct_segment_lengths = len(torch.unique(torch.round(segment_lengths, decimals=10)))
        # print(radius, segment_lengths, 'here')
        # assert len(np.unique(np.round(segment_lengths, 10))) == 1, "All Segments currently must have the same length"
        # assert(np.all(radius < segment_lengths/2)), "Segment approximation radius must be smaller than half of the length of smallest segment"
        batch_normalized_rays = batch_rays / segment_lengths # (B, 2)
        modified_segment_lengths = segment_lengths - (2*radius)

        num_circles_per_segment = torch.ceil(torch.round((modified_segment_lengths / (2*radius)), decimals=10)).type(torch.int32)
        max_num_circles = math.ceil(torch.max(num_circles_per_segment)) + 1

        circle_start_points = start_points + (batch_normalized_rays * radius)

        gaps = (modified_segment_lengths / num_circles_per_segment)

        batch_scaled_rays = (batch_normalized_rays * gaps.view(-1, 1)).view(-1, 1, 2)
        repeated_rays = batch_scaled_rays.repeat(1, max_num_circles, 1)
        repeated_rays[:, 0, :] = 0

        trajectories = torch.cumsum(repeated_rays, dim=1) + circle_start_points.view(-1, 1, 2)

        if isinstance(radius, float):
            shaped_radius = torch.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        elif isinstance(radius, torch.Tensor):
            shaped_radius = torch.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius.view(-1, 1, 1)

        circle_center_radius_pairs = torch.cat((trajectories, shaped_radius), dim=2)

        if num_distinct_segment_lengths == 1:
            circle_center_radius_pairs = circle_center_radius_pairs.view(-1, 3)
        else:
            num_circles_per_segment = num_circles_per_segment.squeeze()
            circle_center_radius_pairs = torch.vstack([circle_center_radius_pairs[i, :(num_circles+1)] for i, num_circles in enumerate(num_circles_per_segment)])

        return circle_center_radius_pairs

    def points_to_circles(self, points):
        # points: (B, 2)
        radii = torch.zeros((points.shape[0], 1)) # TODO
        zero_radius_circles = torch.cat((points, radii), dim=1)
        return zero_radius_circles

    def circles_to_validity(self, obstacle_circles, robot_circles):
        B = robot_circles.shape[0]
        robot_xy = robot_circles[:, :, :2]
        obst_xy = obstacle_circles[:, :2]
        print(robot_xy.device, obst_xy.device)
        distance_mat = torch.sqrt(torch.sum(robot_xy**2, dim=2, keepdim=True) + torch.sum(obst_xy**2, dim=1, keepdim=True).T + (-2 * (robot_xy @ obst_xy.T))) # TODO

        min_dists = robot_circles[:, :, 2].view(B, -1, 1) + obstacle_circles[:, 2].view(1, 1, -1)

        validity_mask = distance_mat > min_dists
        validity_mask = validity_mask.view(B, -1)
        validities = torch.all(validity_mask, dim=1) # TODO

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
        stacked_validities = torch.hstack(stacked_validities) # TODO
        return (stacked_validities.numpy())

    def draw_state(self, ax, state):
        start_time = time.time()
        circles = self.states_to_circles(np.array([self.get_state_value(state)]))[0]
        patch_list = [patches.Circle((x,y), r) for (x,y,r) in circles]
        patch_collection = PatchCollection(patch_list, color='red')
        ax.add_collection(patch_collection)

    def draw_environment(self, ax):
        ax.set_xlim(self.space.x_range[0], self.space.x_range[1])
        ax.set_ylim(self.space.y_range[0], self.space.y_range[1])
        circles = self.space_to_circles()

        patch_list = [patches.Circle((x,y), r) for (x,y,r) in circles]
        patch_collection = PatchCollection(patch_list, color='blue')
        ax.add_collection(patch_collection)
    
    def sample_point(self):
        return self.space.sample_point()
    
    def is_valid(self, state):
        return self.space.is_valid(state)
    
    def make_state(self, state):
        return self.space.make_state(state)
    
    def batch_sample_points_around_target(self, targets):
        return self.space.batch_sample_points_around_target(targets)
if __name__ == "__main__":
    np.random.seed(0)

    env = PolygonalRobot()
    env.set_obstacles(TestSet())
    # env.set_obstacles(NonRegularPolygonObst())
    state = env.make_state(np.array([-1.0,3.0,np.pi/4]))
    state = env.make_state(np.array([-4.0,3.0,np.pi/4]))
    env = ApproximationSpaceTorch(env, do_overapproximation=True)
    env.draw_environment(plt.gca())
    # env.space.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    # env.space.draw_state(plt.gca(), state)
    plt.show()
