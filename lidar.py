import numpy as np 
from space import PointRobot
from obstacle_sets import BiasedPassage, RandomSamplePassage, WeavingPassage
from utils import interpolate_edge, batch_interpolate_edge
import matplotlib.pyplot as plt
import time
from utils import issue_warning
from circle_approximation import ApproximationSpace

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


    def shoot_ray_deterministic(self, sensor_position, interest_point):
        sensor_position = self.engine.get_state_value(sensor_position)
        interest_point = self.engine.get_state_value(interest_point)

        edge_states = interpolate_edge(self.make_state(sensor_position), self.make_state(interest_point), self.edge_validity_delta)
        
        for state in edge_states[1:]:
            if not self.is_valid(state):
                return self.make_state(prev_state)
            prev_state = state
        return self.make_state(prev_state)
    
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



            # self.engine.shoot_ray()
            # points = self.engine.shoot_ray(self.engine.make_state(sensor_position), self.engine.make_state(np.array([ex,ey])), self.max_dist)
            points = interpolate_edge(self.engine.make_state(sensor_position), self.engine.make_state(np.array([ex,ey])), self.resolution)

            # self.engine.draw_environment(plt.gca())
            # plt.scatter(x=[5.0], y=[5.0], color='green', marker='*')
            # plt.scatter(points[:, 0], points[:, 1], color='blue')
            # plt.show()
            

            obstacle_point = None
            for state in points[1:]:
                if not self.engine.is_valid(state):
                    obstacle_point = self.engine.make_state(state)
                    break
            # print(type(sensor_position), )
            last_point = points[-1]
            angle_noise = np.random.normal(loc=0, scale=self.noise[0])
            dist_noise = np.random.normal(loc=0, scale=self.noise[1])
            if obstacle_point:
                readings.append((angle+angle_noise, obstacle_point, self.engine.dist(self.engine.make_state(sensor_position), obstacle_point)+dist_noise, self.engine.make_state(last_point)))
            else:
                readings.append((angle+angle_noise, obstacle_point, np.inf, self.engine.make_state(last_point)))


        return readings
        
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
        # pts = pts.reshape(B, -1)
        # Loop and find the first instance of invalid:
        # edge_validities = np.array([np.all(pt_validities[i, :steps[i]]) for i in range(len(steps))])
        # [pt_validities[i, :steps[i]] for i in range(len(steps))]

        for i in range(len(steps)):
            idx_list = np.where(pt_validities[i, :steps[i]] == False)[0]
            if len(idx_list) > 0:
                interest_idx = idx_list[0]
                obstacle_point = pts[i, :steps[i]][interest_idx]
                readings.append((angles[i], self.engine.make_state(obstacle_point), self.engine.dist(self.engine.make_state(sensor_position), self.engine.make_state(obstacle_point)), self.engine.make_state(pts[i, :steps[i]][-1])))
            else:
                readings.append((angles[i], None, np.inf, self.engine.make_state(pts[i, :steps[i]][-1])))

        return readings



if __name__ == '__main__':
    lidar = OptimizedLidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))
    # lidar = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))
    # lidar = Lidar(0, (0, 2*np.pi), 100, 4.9, RandomSamplePassage(num_walls=3))
    # lidar = Lidar()
    # lidar = Lidar(0,0,0,0)

    st = time.time()
    readings = lidar.read_sensor(np.array((5.0,5.0)))
    et = time.time()
    print(f"Time to Run: {et-st}")

    # print(readings)

    # for r in readings:
    #     if r[1]:
    #         print(r[0], r[1].value, r[2])
    #     else:
    #         print(r[0], r[1], r[2])

    lidar.engine.draw_environment(plt.gca())
    plt.scatter(x=[5.0], y=[5.0], color='green', marker='*')
    plt.show()

    lidar.engine.draw_environment(plt.gca())
    plt.scatter(x=[5.0], y=[5.0], color='green', marker='*')

    # print([r[1].value for r in readings])
    lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
    plt.scatter(lidar_points[:, 0], lidar_points[:, 1], color='red', zorder=2)
    # print(lidar_points)

    

    plt.show()

    locs = []
    all_lidar_points = []
    print("Before loops")
    for i in range(10):
        print(f"Running Iteration: {i}")
        loc = lidar.engine.sample_valid_point()
        readings = lidar.read_sensor(loc)
        lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
        locs.append(loc.value)

        all_lidar_points.append(lidar_points)

    all_lidar_points = np.vstack(all_lidar_points)
    locs = np.array(locs)

    lidar.engine.draw_environment(plt.gca())
    plt.scatter(x=locs[:, 0], y=locs[:, 1], color='green', marker='*')

    # lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
    plt.scatter(all_lidar_points[:, 0], all_lidar_points[:, 1], color='red', zorder=2)
    plt.show()


