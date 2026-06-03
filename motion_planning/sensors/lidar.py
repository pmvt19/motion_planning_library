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


class Lidar():
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
    
    def read_sensor(self, sensor_position):

        sensor_position = self.engine.get_state_value(sensor_position)

        readings = [] # Format: [(angle, point, dist)]
        angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_angles)

        for angle in angles:
            # print(angle)

            dx = np.cos(angle) * self.max_dist
            dy = np.sin(angle) * self.max_dist

            sx, sy = sensor_position
            
            ex, ey = sx + dx, sy + dy

            points = interpolate_edge(self.engine.make_state(sensor_position), self.engine.make_state(np.array([ex,ey])), self.resolution)
            

            obstacle_point = None
            for state in points[1:]:
                if not self.engine.is_valid(state):
                    obstacle_point = self.engine.make_state(state)
                    break

            last_point = points[-1]

            # angle_noise = np.random.normal(loc=0, scale=self.noise[0])
            # dist_noise = np.random.normal(loc=0, scale=self.noise[1])

            angle_noise = 0
            dist_noise = 0
            if obstacle_point:
                readings.append((angle+angle_noise, obstacle_point, self.engine.dist(self.engine.make_state(sensor_position), obstacle_point)+dist_noise, self.engine.make_state(last_point)))
            else:
                readings.append((angle+angle_noise, obstacle_point, np.inf, self.engine.make_state(last_point)))

        # TODO: Change the return result to only angle, dist

        return readings

if __name__ == '__main__':
    # np.random.seed(0)
    # lidar = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))
    # lidar = Lidar(0, (0, 2*np.pi), 100, 4.9, RandomSamplePassage(num_walls=3))
    # lidar = Lidar()
    lidar = Lidar(0,0,0,0)

    st = time.time()
    readings = lidar.read_sensor(np.array([5.0, 5.0]))

    et = time.time()
    print(f"Time to Run: {et-st}")

    lidar.engine.draw_environment(plt.gca())
    plt.scatter(x=[5.0], y=[5.0], color='green', marker='*')
    plt.show()

    lidar.engine.draw_environment(plt.gca())
    plt.scatter(x=[5.0], y=[5.0], color='green', marker='*')

    lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
    plt.scatter(lidar_points[:, 0], lidar_points[:, 1], color='red', zorder=2)
    plt.show()


    locs = []
    all_lidar_points = []
    st = time.time()
    print("Before loops")
    for i in range(10):
        print(f"Running Iteration: {i}")
        loc = lidar.engine.sample_valid_point()
        readings = lidar.read_sensor(loc)
        lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
        locs.append(loc.value)

        all_lidar_points.append(lidar_points.reshape(-1, 2))
    et = time.time()
    print(f"Time to Run All Points: {et-st}")

    all_lidar_points = np.vstack(all_lidar_points)
    locs = np.array(locs)

    lidar.engine.draw_environment(plt.gca())
    plt.scatter(x=locs[:, 0], y=locs[:, 1], color='green', marker='*')

    plt.scatter(all_lidar_points[:, 0], all_lidar_points[:, 1], color='red', zorder=2)
    plt.show()


