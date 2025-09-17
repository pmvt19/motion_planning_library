import numpy as np 
from shapely import Polygon
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


class SuperOptimizedLidar():
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
        # TODO: Should be doable with fully parallelized numpy operations

        sensor_position = self.engine.get_state_value(sensor_position)

        readings = [] # Format: [(angle, point, dist)]
        angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_angles)

        for angle in angles:
            # print(angle)

            dx = np.cos(angle) * self.max_dist
            dy = np.sin(angle) * self.max_dist

            sx, sy = sensor_position
            
            ex, ey = sx + dx, sy + dy

            max_dist_point = np.array([ex,ey])
            farthest_max_dist_point = np.array([ex,ey])
            min_dist = self.max_dist

            for line in self.lines:
                x1, y1, x2, y2 = line

                x3, y3 = sensor_position
                x4, y4 = max_dist_point

                a = (x4 - x3) * (y3 - y1) - (y4 - y3) * (x3 - x1)
                b = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1)
                c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

                alpha = a / b
                beta = c / b

                if np.isclose(b, 0): # Two Line Segments are Parallel
                    pass
                elif np.isclose(a, 0) and np.isclose(b, 0): # Lines are Colinear (Need to deal with edge case though)
                    print("Found Super Rare Edge Case: Not Implemented")
                    raise NotImplementedError
                elif 0 < alpha and alpha < 1 and 0 < beta and beta < 1:
                    # do something
                    x0 = x1 + alpha * (x2 - x1)
                    y0 = y1 + alpha * (y2 - y1)
                    my_dist = self.engine.dist(self.engine.make_state(np.array([x0, y0])), self.engine.make_state(sensor_position))
                    if my_dist < min_dist:
                        max_dist_point = np.array([x0, y0])
                        min_dist = my_dist
                else:
                    pass

            if min_dist < self.max_dist:
                readings.append((angle, self.engine.make_state(max_dist_point), min_dist, self.engine.make_state(farthest_max_dist_point)))
            else:
                readings.append((angle, None, np.inf, self.engine.make_state(farthest_max_dist_point)))

        return readings

    def read_sensor_optimized(self, sensor_position):
        # TODO: Should be doable with fully parallelized numpy operations

        sensor_position = self.engine.get_state_value(sensor_position)

        readings = [] # Format: [(angle, point, dist)]
        angles = np.linspace(self.angle_range[0], self.angle_range[1], self.num_angles)

        dx_cos = np.cos(angles).reshape(-1, 1) * self.max_dist
        dy_sin = np.sin(angles).reshape(-1, 1) * self.max_dist

        farthest_points = sensor_position.reshape(-1, 2) * np.hstack((dx_cos, dy_sin)) 
        sensor_position_repeated = np.repeat(sensor_position.reshape(1, -1), self.num_angles, axis=0)

        print(self.lines.shape)

        x1s = self.lines[:, 0].reshape(-1, 1) # (L, 1)
        y1s = self.lines[:, 1].reshape(-1, 1) # (L, 1)
        x2s = self.lines[:, 2].reshape(-1, 1) # (L, 1)
        y2s = self.lines[:, 3].reshape(-1, 1) # (L, 1)

        # x3, y3 = sensor_position
        x3s = sensor_position[0] # (1,)
        y3s = sensor_position[1] # (1,)

        # x3s = sensor_position_repeated[:, 0].reshape(-1, 1)
        # y3s = sensor_position_repeated[:, 1].reshape(-1, 1)

        x4s = farthest_points[:, 0].reshape(-1, 1) # (self.num_angles, 1)
        y4s = farthest_points[:, 1].reshape(-1, 1) # (self.num_angles, 1)

        # a_s = (x4s - x3s) * (y3s - y1s) - (y4s - y3s) * (x3s - x1s) # (self.num_angles, L)

        temp = (x4s - x3s)
        temp2 = (y3s - y1s)

        temp3 = temp * temp2.T
        print(x4s.shape, x3s.shape, temp.shape)
        print(y3s.shape, y1s.shape, temp2.shape)

        print(temp.shape, temp2.shape, temp3.shape)
        print("-----")

        temp4 = (y4s - y3s)
        temp5 = (x3s - x1s)
        
        print(y4s.shape, y3s.shape, temp4.shape)
        print(x3s.shape, x1s.shape, temp5.shape)

        temp6 = temp4 * temp5.T
        print(temp4.shape, temp5.shape, temp6.shape)

        print("-----")

        a_s = (x4s - x3s) * (y3s - y1s).T - (y4s - y3s) * (x3s - x1s).T # (self.num_angles, L)
        b_s = (x4s - x3s) * (y2s - y1s).T - (y4s - y3s) * (x2s - x1s).T # (self.num_angles, L)
        # c_s = (x2s - x1s) * (y3s - y1s).T - (y2s - y1s) * (x3s - x1s).T # (self.num_angles, L)
        # print(a_s.shape, b_s.shape, c_s.shape)
        # print(x2s.shape, x1s.shape)
        t1 = (x2s - x1s)
        t2 = (y3s - y1s)
        t3 = t1 * t2.T
        print(t1.shape, t2.shape, t3.shape)

        exit()
        # a_s = (x4s - x3s) * (y3s - y1s) - (y4s - y3s) * (x3s - x1s) # (self.num_angles, L)
        # b_s = (x4s - x3s) * (y2s - y1s) - (y4s - y3s) * (x2s - x1s) # (self.num_angles, L)
        # c_s = (x2s - x1s) * (y3s - y1s) - (y2s - y1s) * (x3s - x1s) # (self.num_angles, L)

        # alphas = a_s / b_s 
        # betas = c_s / b_s



        for angle in angles:
            # print(angle)

            dx = np.cos(angle) * self.max_dist
            dy = np.sin(angle) * self.max_dist

            sx, sy = sensor_position
            
            ex, ey = sx + dx, sy + dy

            max_dist_point = np.array([ex,ey])
            farthest_max_dist_point = np.array([ex,ey])
            min_dist = self.max_dist

            for line in self.lines:
                x1, y1, x2, y2 = line

                x3, y3 = sensor_position
                x4, y4 = max_dist_point

                a = (x4 - x3) * (y3 - y1) - (y4 - y3) * (x3 - x1)
                b = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1)
                c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

                alpha = a / b
                beta = c / b

                if np.isclose(b, 0): # Two Line Segments are Parallel
                    pass
                elif np.isclose(a, 0) and np.isclose(b, 0): # Lines are Colinear (Need to deal with edge case though)
                    print("Found Super Rare Edge Case: Not Implemented")
                    raise NotImplementedError
                elif 0 < alpha and alpha < 1 and 0 < beta and beta < 1:
                    # do something
                    x0 = x1 + alpha * (x2 - x1)
                    y0 = y1 + alpha * (y2 - y1)
                    my_dist = self.engine.dist(self.engine.make_state(np.array([x0, y0])), self.engine.make_state(sensor_position))
                    if my_dist < min_dist:
                        max_dist_point = np.array([x0, y0])
                        min_dist = my_dist
                else:
                    pass

            if min_dist < self.max_dist:
                readings.append((angle, self.engine.make_state(max_dist_point), min_dist, self.engine.make_state(farthest_max_dist_point)))
            else:
                readings.append((angle, None, np.inf, self.engine.make_state(farthest_max_dist_point)))

        return readings           



    

if __name__ == '__main__':
    lidar = SuperOptimizedLidar(None, (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))
    # lidar = OptimizedLidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))
    # lidar = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))
    # lidar = Lidar(0, (0, 2*np.pi), 100, 4.9, RandomSamplePassage(num_walls=3))
    # lidar = Lidar()
    # lidar = Lidar(0,0,0,0)

    readings = lidar.read_sensor_optimized(np.array([5.0, 5.0]))
    exit()

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

    # lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
    plt.scatter(all_lidar_points[:, 0], all_lidar_points[:, 1], color='red', zorder=2)
    plt.show()


