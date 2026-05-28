import time

import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon

from motion_planning.obstacle_sets import (
    BiasedPassage,
)
from motion_planning.space import ApproximationSpace, PointRobot
from motion_planning.utils import (
    batch_interpolate_edge,
    interpolate_edge,
    issue_warning,
)

class SuperOptimizedLidar():
    def __init__(self, noise, angle_range, num_angles, max_dist, obstacle_set=None, verbose=False):
        self.engine = PointRobot()
        
        issue_warning(True, "Lidar Noise does not work", 'warning')

        self.noise = noise
        self.angle_range = angle_range
        self.num_angles = num_angles
        self.max_dist = max_dist
        self.verbose = verbose

        if obstacle_set:
            self.engine.set_obstacles(obstacle_set)

        self.lines = []
        for obs in obstacle_set.obstacles:
            if isinstance(obs, Polygon):
                x, y = obs.exterior.xy

                for i in range(len(x)-1):
                    self.lines.append([x[i], y[i], x[i+1], y[i+1]])

            else:
                print("Only Polygon Obstacles are Supported")
                raise NotImplementedError
        self.lines = np.array(self.lines)

    def read_sensor(self, sensor_position):
        sensor_position = self.engine.get_state_value(sensor_position)

        readings = [] # Format: [(angle, point, dist)]
        angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_angles)

        dx_cos = np.cos(angles).reshape(-1, 1) * self.max_dist
        dy_sin = np.sin(angles).reshape(-1, 1) * self.max_dist

        farthest_points = sensor_position.reshape(-1, 2) + np.hstack((dx_cos, dy_sin)) 

        x1s = self.lines[:, 0].reshape(-1, 1) # (L, 1)
        y1s = self.lines[:, 1].reshape(-1, 1) # (L, 1)
        x2s = self.lines[:, 2].reshape(-1, 1) # (L, 1)
        y2s = self.lines[:, 3].reshape(-1, 1) # (L, 1)

        x3s = sensor_position[0] # (1,)
        y3s = sensor_position[1] # (1,)

        x4s = farthest_points[:, 0].reshape(-1, 1) # (self.num_angles, 1)
        y4s = farthest_points[:, 1].reshape(-1, 1) # (self.num_angles, 1)

        a_s = (x4s - x3s) * (y3s - y1s).T - (y4s - y3s) * (x3s - x1s).T # (self.num_angles, L)
        b_s = (x4s - x3s) * (y2s - y1s).T - (y4s - y3s) * (x2s - x1s).T # (self.num_angles, L)
        c_s = ((x2s - x1s) * (y3s - y1s) - (y2s - y1s) * (x3s - x1s)).T # (1, L)

        alphas = a_s / b_s 
        betas = c_s / b_s

        x0s = (x1s.T + alphas * (x2s - x1s).T)
        y0s = (y1s.T + alphas * (y2s - y1s).T)


        points = np.stack((x0s, y0s), axis=2)

        sensor_position_shaped = sensor_position.reshape(1, -1)
        dists = np.sum(points**2, axis=2, keepdims=True) + np.sum(sensor_position_shaped**2, axis=1, keepdims=True).T + (-2 * (points @ sensor_position_shaped.T))
        dists = dists.squeeze()

        b_s_mask = np.isclose(b_s, 0)
        dists[b_s_mask] = np.inf
        
        alphas_low_mask = alphas < 0
        dists[alphas_low_mask] = np.inf         

        alphas_high_mask = alphas > 1
        dists[alphas_high_mask] = np.inf 

        betas_low_mask = betas < 0
        dists[betas_low_mask] = np.inf 

        betas_high_mask = betas > 1
        dists[betas_high_mask] = np.inf 

    
        if self.verbose:
            print(f"Mask Effectiveness (b_s_mask): {np.sum(b_s_mask)}")
            print(f"Mask Effectiveness (alphas_low_mask): {np.sum(alphas_low_mask)}")
            print(f"Mask Effectiveness (alphas_high_mask): {np.sum(alphas_high_mask)}")
            print(f"Mask Effectiveness (betas_low_mask): {np.sum(betas_low_mask)}")
            print(f"Mask Effectiveness (betas_high_mask): {np.sum(betas_high_mask)}")

        dists = np.sqrt(dists)
        dists[dists > self.max_dist] = np.inf

        
        min_idxes = np.argmin(dists, axis=1)
        min_vals = np.min(dists, axis=1)

        masking = min_vals < np.inf

        for i in range(len(min_vals)):
            fp = farthest_points[i]
            if masking[i]:
                line_idx = min_idxes[i]
                intersection_point = points[i, line_idx]
                my_dist = min_vals[i]
                readings.append((angles[i], self.engine.make_state(intersection_point), my_dist, self.engine.make_state(fp)))
            else:
                readings.append((angles[i], None, np.inf, self.engine.make_state(fp)))

        return readings        