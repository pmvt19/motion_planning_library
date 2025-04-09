from space import RobotSpace, PlanarMobileArm
from obstacle_sets import TestSet
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point, Polygon
from sklearn.metrics import pairwise_distances
import time

class ApproximationSpace(RobotSpace):
    def __init__(self, space : RobotSpace, do_overapproximation=False):
        super().__init__()
        self.space = space
        self.do_overapproximation = do_overapproximation
        self.obstacle_circles = self.space_to_circles()
    
    def obstacles_to_aabb(self, obstacles):
        aabbs = []
        for obs in obstacles:
            xs, ys = obs.exterior.xy
            x = np.min(xs)
            y = np.min(ys)
            w = np.max(xs) - x
            h = np.max(ys) - y

            aabbs.append([x+(w/2), y+(h/2), w, h])
        return np.array(aabbs)

    def space_to_circles(self):
        aabbs = self.obstacles_to_aabb(self.space.obstacles)
        obst_circles = self.rectangles_to_circles(aabbs)
        return obst_circles
    
    def states_to_circles(self, states):
        # States : (B, d)
        representations = self.space.batch_get_robot_representations(states)
        B, *_ = states.shape
        rect_circles = self.rectangles_to_circles(representations['rectangles']).reshape(B, -1, 3)
        seg_circles = self.segments_to_circles(representations['segments'], 0.1).reshape(B, -1, 3)
        point_circles = self.points_to_circles(representations['points']).reshape(B, -1, 3)
        state_circles = np.concatenate((rect_circles, seg_circles, point_circles), axis=1)
        return state_circles

    def rectangles_to_circles(self, aa_rect):
        # Let's say that we get the output of this as (x,y,w,h) -> Thus would have a shape of (N,4)
        # The way I was thinking about it, one would need to loop through each rectangle to create the points
        # While this isn't necessarily a problem for the environment (since we can just create this once and leave it)
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
    
    def segments_to_circles(self, segments : np.ndarray, radius : float):
        """
            B : Number of entries in the batch
            2 : This value is fixed at two since a segment must have only 2 end points
            2 : Dimension of the segment
        """
        # segments : (B, 2, 2)

        start_points = segments[:, 0, :] # (B, 2)
        end_points = segments[:, 1, :] # (B, 2)

        batch_rays = end_points - start_points
        segment_lengths = np.linalg.norm(batch_rays, axis=1).reshape(-1, 1) # (B,1)

        num_distinct_segment_lengths = len(np.unique(np.round(segment_lengths, 10)))
        # assert len(np.unique(np.round(segment_lengths, 10))) == 1, "All Segments currently must have the same length"
        assert(np.all(radius < segment_lengths/2)), "Segment approximation radius must be smaller than half of the length of smallest segment"
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

        shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)

        if num_distinct_segment_lengths == 1:
            circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 3)
        else:
            num_circles_per_segment = num_circles_per_segment.squeeze()
            circle_center_radius_pairs = np.vstack([circle_center_radius_pairs[i, :(num_circles+1)] for i, num_circles in enumerate(num_circles_per_segment)])

        return circle_center_radius_pairs

    def points_to_circles(self, points):
        # points: (B, 2)
        radii = np.zeros((points.shape[0], 1))
        zero_radius_circles = np.concatenate((points, radii), axis=1)
        return zero_radius_circles

    def circles_to_validity(self, obstacle_circles, robot_circles):
        B = robot_circles.shape[0]
        # print(obstacle_circles.shape, robot_circles.shape)
        robot_xy = robot_circles[:, :, :2]
        obst_xy = obstacle_circles[:, :2]
        distance_mat = np.sqrt(np.sum(robot_xy**2, axis=2, keepdims=True) + np.sum(obst_xy**2, axis=1, keepdims=True).T + (-2 * (robot_xy @ obst_xy.T)))

        min_dists = robot_circles[:, :, 2].reshape(B, -1, 1) + obstacle_circles[:, 2].reshape(1, 1, -1)

        validity_mask = distance_mat > min_dists
        validity_mask = validity_mask.reshape(B, -1)
        validities = np.all(validity_mask, axis=1)

        return validities
    
    def batch_is_valid(self, states):
        robot_circles = self.states_to_circles(states)
        return self.circles_to_validity(self.obstacle_circles, robot_circles)

    def draw_state(self, ax, state):
        circles = self.states_to_circles(np.array([state]))[0]
        # print(circles.shape)
        for i, (x, y, r) in enumerate(circles):
            point = Point(x, y)
            c = point.buffer(r)
            x, y = c.exterior.xy
            ax.fill(x, y, 'red')

    def draw_environment(self, ax):
        circles = self.space_to_circles()
        for i, (x, y, r) in enumerate(circles):
            point = Point(x, y)
            c = point.buffer(r)
            x, y = c.exterior.xy
            ax.fill(x, y, 'blue')
    
    def sample_point(self):
        return self.space.sample_point()
    
    def is_valid(self, state):
        return self.space.is_valid(state)
    
    def make_state(self, state):
        return self.space.make_state(state)
    
if __name__ == "__main__":
    np.random.seed(0)
    # env = PlanarMobileArm(num_links=3, arm_lengths=[1, 1, 1])
    env = PlanarMobileArm(num_links=3, arm_lengths=[1, 1, 1])
    env.set_obstacles(TestSet())

    space = ApproximationSpace(env)
    obst_circles = space.space_to_circles()

    N = 2000
    numpystates = [env.sample_point() for _ in range(N)]
    states = np.array([state.value for state in numpystates])
    
    # start_time = time.time()
    # traditional_validities = [env.is_valid(state) for state in numpystates]
    # end_time = time.time()
    # print(f"Traditional is_valid Time: {end_time-start_time}")

    # start_time = time.time()
    # space.batch_is_valid(states)
    # end_time = time.time()
    # print(f"Batched is_valid Time: {end_time-start_time}")

    # validities = space.batch_is_valid(states)
    # print(obst_circles.shape)
    # for i in range(N):
    #     plt.clf()
    #     plt.title(validities[i])
    #     # env.draw_environment(plt.gca())
    #     space.draw_environment(plt.gca())
        
    #     # env.draw_state(plt.gca(), numpystates[i])
    #     space.draw_state(plt.gca(), numpystates[i].value)
    #     plt.show()

    ## Batch get edge states ##
    start_states = states[:(N//2)]
    end_states = states[(N//2):]
    print(f"Validating {(N//2)} edges")
    start_time = time.time()
    edge_validities = space.batch_is_valid_edge(start_states, end_states)
    end_time = time.time()
    print(f"Time to batch validate edges: {end_time - start_time}")

    start_time = time.time()
    validities_edge = [env.is_valid_edge(start_states[i], end_states[i]) for i in range((N//2))]
    end_time = time.time()
    print(f"Time to unbatched validate edges: {end_time - start_time}")

    

    # print(start_states.shape, end_states.shape)
    # pts, steps = env.batch_get_edge_states(start_states, end_states)
    ## Batch get edge states ##
    print("Unbatched function")
    # print(env.get_edge_states(start_states[0], end_states[0]))
    # print(env.get_edge_states(start_states[1], end_states[1]))

    # edge_states = [pts[i, :steps[i], :] for i in range(len(steps))]

    # B, d = (N//2), 5
    # old_plts = pts
    # pts = pts.reshape(-1, d)
    # pt_validities = space.batch_is_valid(pts).reshape(B, -1)
    # edge_validities = [np.all(pt_validities[i, :steps[i]]) for i in range(len(steps))]

    # for i in range(B):
    #     plt.clf()
    #     plt.title(edge_validities[i])
    #     space.draw_environment(plt.gca())
    #     for state in old_plts[i, :steps[i], :]:
    #         space.draw_state(plt.gca(), state)
    #     plt.show()


    exit()

    # N = 10000
    # aa_rect = np.array([
    #         [0, 0, 8, 5.0] for _ in range(N)
    #     ])
    
    # print(aa_rect.shape)
    # start_time = time.time()
    # space.rectangles_to_circles(aa_rect)
    # end_time = time.time()
    # print(f"Rectangles to Circles Time: {end_time-start_time}")
    
    
    # aa_rect_robot = np.array([[0, 5, 1, 1.5]])
    # obst_circles = space.rectangles_to_circles(aa_rect)
    # robot_circles = space.rectangles_to_circles(aa_rect_robot)
    # space.draw_env(plt.gca(), obst_circles, aa_rect)
    # space.draw_env(plt.gca(), robot_circles, aa_rect_robot)
    # plt.show()

    # print("obstacle circles")
    # print(obst_circles)
    # print("robot circles")
    # print(robot_circles)

    # dists = pairwise_distances(obst_circles[:, :2], robot_circles[:, :2])
    # print(dists)
    # min_dists = obst_circles[:, 2].reshape(-1, 1) + robot_circles[:, 2].reshape(1, -1)
    # print(min_dists)

    # is_valid_per_circle = dists > min_dists
    # print(np.all(is_valid_per_circle, axis=0))

    # # ### TEST SEGMENTS TO CIRCLES ###

    # N = 20000
    # numpystates = [env.sample_point() for _ in range(N)]
    # states = np.array([state.value for state in numpystates])
    # segment_points = env.batch_forward_kinematics(states)#.reshape(-1, 2, 2)
    # start_points = segment_points[:, :-1, :]
    # end_points = segment_points[:, 1:, :]
    # segment_start_end_points = np.concatenate((start_points, end_points), axis=2).reshape(-1, 4).reshape(-1, 2, 2)

    # # start_time = time.time()
    # # segment_circles = space.segments_to_circles(segment_start_end_points, radius=0.04)
    # # end_time = time.time()
    # # print(f"Segments to Circles Time: {end_time - start_time}")

    # start_time = time.time()
    # env.batch_get_robot_representations(states)
    # end_time = time.time()

    # print(f"States to Representations Time: {end_time - start_time}")

    # space.states_to_circles(states)

    # # ### TEST SEGMENTS TO CIRCLES ###

    # for i in range(N):
    #     plt.clf()
    #     env.draw_state(plt.gca(), numpystates[i])
    #     space.draw_env(plt.gca(), segment_circles, [])
    #     plt.show()

    