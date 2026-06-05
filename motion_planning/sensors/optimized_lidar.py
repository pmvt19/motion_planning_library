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


class OptimizedLidar():
    def __init__(self, noise, angle_range, num_angles, max_dist, obstacle_set=None):
        self.engine = PointRobot()
        
        issue_warning(True, "Lidar Noise does not work", 'warning')

        self.noise = noise
        self.angle_range = angle_range
        self.num_angles = num_angles
        self.max_dist = max_dist
        self.resolution = 0.05

        if obstacle_set:
            self.engine.set_obstacles(obstacle_set)
        self.engine = ApproximationSpace(self.engine, batch_size=1000, do_overapproximation=True)

    def read_sensor(self, sensor_position):
        sensor_position = self.engine.get_state_value(sensor_position)

        readings = [] # Format: [(angle, point, dist)]
        angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_angles)

        dx_cos = np.cos(angles).reshape(-1, 1) * self.max_dist
        dy_sin = np.sin(angles).reshape(-1, 1) * self.max_dist

        farthest_points = sensor_position.reshape(-1, 2) * np.hstack((dx_cos, dy_sin)) 
        sensor_position_repeated = np.repeat(sensor_position.reshape(1, -1), self.num_angles, axis=0)

        B, d = farthest_points.shape

        pts, steps = batch_interpolate_edge(sensor_position_repeated, farthest_points, self.resolution, None)
        pts_reshape = pts.reshape(-1, d)
        pt_validities = self.engine.batch_is_valid(pts_reshape).reshape(B, -1)

        for i in range(len(steps)):
            idx_list = np.where(pt_validities[i, :steps[i]] == False)[0]
            if len(idx_list) > 0:
                interest_idx = idx_list[0]
                obstacle_point = pts[i, :steps[i]][interest_idx]
                readings.append((angles[i], self.engine.make_state(obstacle_point), self.engine.dist(self.engine.make_state(sensor_position), self.engine.make_state(obstacle_point)), self.engine.make_state(pts[i, :steps[i]][-1])))
            else:
                readings.append((angles[i], None, np.inf, self.engine.make_state(pts[i, :steps[i]][-1])))

        return readings