from space import RobotSpace, PlanarMobileArm
from obstacle_sets import TestSet
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point, Polygon
from sklearn.metrics import pairwise_distances
import time

def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return np.stack([x1, y1, x2, y2], axis=1)

def xywh_to_corners(boxes):
    boxes = np.asarray(boxes)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # Shape (N, 4, 2): 4 corners per box
    corners = np.stack([
        np.stack([x1, y1], axis=1),  # top-left
        np.stack([x2, y1], axis=1),  # top-right
        np.stack([x2, y2], axis=1),  # bottom-right
        np.stack([x1, y2], axis=1),  # bottom-left
    ], axis=1)

    return corners

def xywh_center_to_corners(boxes):
    boxes = np.asarray(boxes)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    corners = np.stack([
        np.stack([x1, y1], axis=1),  # top-left
        np.stack([x2, y1], axis=1),  # top-right
        np.stack([x2, y2], axis=1),  # bottom-right
        np.stack([x1, y2], axis=1),  # bottom-left
    ], axis=1)

    return corners

class ApproximationSpace():
    def __init__(self, space : RobotSpace):
        self.space = space

    def space_to_circles(self):
        raise NotImplementedError

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

        print(aa_rect.shape)
        N = len(aa_rect)
        circles = []
        for i in range(N):
            x, y, w, h = aa_rect[i]
            r = min(w, h)/2
            # inflated_r = r * math.sqrt(2)
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

            # (B, x, y, r)
        
        print(circles)
        circles = np.array(circles)
        return circles
        # self.draw_env(circles, aa_rect)

        # raise NotImplementedError
    
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
        assert len(np.unique(np.round(segment_lengths, 10))) == 1, "All Segments currently must have the same length"
        assert(np.all(radius < segment_lengths/2)), "Segment approximation radius must be smaller than half of the length of smallest segment"
        batch_normalized_rays = batch_rays / segment_lengths # (B, 2)
        modified_segment_lengths = segment_lengths - (2*radius)


        num_circles_per_segment = np.ceil(np.round((modified_segment_lengths / (2*radius)), 10))
        max_num_circles = math.ceil(np.max(num_circles_per_segment)) + 1

        circle_start_points = start_points + (batch_normalized_rays * radius)

        gaps = (modified_segment_lengths / num_circles_per_segment)

        batch_scaled_rays = (batch_normalized_rays * gaps.reshape(-1, 1)).reshape(-1, 1, 2)
        repeated_rays = np.repeat(batch_scaled_rays, (max_num_circles), axis=1)
        repeated_rays[:, 0, :] = 0

        trajectories = np.cumsum(repeated_rays, axis=1) + circle_start_points.reshape(-1, 1, 2)

        shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)

        circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 3)
        return circle_center_radius_pairs


    def points_to_circles(self):
        raise NotImplementedError
    
    def draw_env(self, ax, circles, rectangles):
        # plt.figure()
        # plt.plot()
        
        for i, r in enumerate(rectangles):
            corners = xywh_center_to_corners(rectangles)[i]
            print(corners, "here", rectangles)
            shape = Polygon(corners)
            ax.plot(*shape.exterior.xy, color='black')
        for i, (x, y, r) in enumerate(circles):
            # print(x, y, r)
            point = Point(x, y)
            c = point.buffer(r)
            x, y = c.exterior.xy
            ax.fill(x, y, 'blue')  # 'g-' for green line
            
            # if i >= 5:
            #     break
        # plt.show()





    
if __name__ == "__main__":
    np.random.seed(0)
    # env = PlanarMobileArm(num_links=3, arm_lengths=[1, 1, 1])
    env = PlanarMobileArm(num_links=3, arm_lengths=[1, 1, 1])
    # env.set_obstacles(TestSet())

    space = ApproximationSpace(env)

    # aa_rect = np.array([
    #         [0, 0, 8, 5.0],
    #         [3, 4, 3, 3],
    #     ])
    
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
    N = 3
    numpystates = [env.sample_point() for _ in range(N)]
    states = np.array([state.value for state in numpystates])
    segment_points = env.batch_forward_kinematics(states)#.reshape(-1, 2, 2)
    start_points = segment_points[:, :-1, :]
    end_points = segment_points[:, 1:, :]
    segment_start_end_points = np.concatenate((start_points, end_points), axis=2).reshape(-1, 4).reshape(-1, 2, 2)
    segment_circles = space.segments_to_circles(segment_start_end_points, radius=0.1)

    for i in range(N):
        plt.clf()
        env.draw_state(plt.gca(), numpystates[i])
        space.draw_env(plt.gca(), segment_circles, [])
        plt.show()

    