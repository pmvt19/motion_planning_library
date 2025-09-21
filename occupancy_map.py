import math
import numpy as np
import matplotlib.pyplot as plt

from lidar import Lidar
from obstacle_sets import BiasedPassage, RandomSamplePassage, ParkingSpace
from space import PointRobot

from scipy.signal import convolve2d
from heapq import *

def line_seg_to_points_dist(p1: np.ndarray, p2: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute the shortest distance between a line segment (p1, p2) and a set of points.

    Parameters
    ----------
    p1 : np.ndarray
        Starting point of the line segment, shape (d,)
    p2 : np.ndarray
        Ending point of the line segment, shape (d,)
    points : np.ndarray
        Array of points, shape (N, d)

    Returns
    -------
    np.ndarray
        Distances from each point to the line segment, shape (N,)
    """
    # Vector along the line segment
    seg_vec = p2 - p1
    seg_len_sq = np.dot(seg_vec, seg_vec)

    # Vectors from p1 to the points
    p1_to_points = points - p1

    # Project each point onto the line, normalized by segment length
    t = np.einsum('ij,j->i', p1_to_points, seg_vec) / seg_len_sq

    # Clamp t to [0,1] to stay within the segment
    t = np.clip(t, 0.0, 1.0)

    # Closest point on the segment for each point
    proj_points = p1 + np.outer(t, seg_vec)

    # Distances to the closest points
    dists = np.linalg.norm(points - proj_points, axis=1)

    return dists

# TODO: CHANGE EMPTY TO 0.1?
# EMPTY = 0.1
EMPTY = 0.01
SOFT_BUFFER = 0.4
UNKNOWN = 0.5
HARD_BUFFER = 0.9
OCCUPIED = 1
PATH = 1.5

class OccupancyMap():
    def __init__(self, resolution=0.1):

        

        self.res = resolution

        # self.x_range = [-10,10]
        # self.y_range = [-10,10]

        self.x_range = [0, 30]
        self.y_range = [0, 10]

        # self.x_range = [-15,15]
        # self.y_range = [-15,15]

        self.x_points = np.arange(self.x_range[0]-resolution, self.x_range[1]+resolution, resolution)
        self.y_points = np.arange(self.y_range[0]-resolution, self.y_range[1]+resolution, resolution)

        # print(self.x_points)
        # print(self.y_points)

        

        self.x_points_centered = self.x_points + resolution/2
        self.y_points_centered = self.y_points + resolution/2

        # print(self.x_points_centered)
        # print(self.y_points_centered)

        self.x_idxes = (self.x_points - self.x_range[0]) / self.res
        self.y_idxes = (self.y_points - self.y_range[0]) / self.res

        x_ind, y_ind = np.meshgrid(np.round(self.x_idxes, 0).astype(np.int32), np.round(self.y_idxes, 0).astype(np.int32))
        self.inds = np.stack((x_ind, y_ind), axis=2)
        self.inds = self.inds.reshape(-1, 2)

        # print(self.x_idxes)
        # print(self.y_idxes)

        # self.map = np.zeros((len(self.x_points), len(self.y_points)))
        self.map = np.ones((len(self.x_points), len(self.y_points))) * UNKNOWN

        x_cir, y_cir = np.meshgrid(self.x_points_centered, self.y_points_centered)

        self.circles = np.stack((x_cir, y_cir), axis=2)
        self.circles = self.circles.reshape(-1, 2)
        self.circles = np.hstack((self.circles, np.ones((len(self.circles), 1)) * (self.res/2)))

        # print(self.circles.shape)
        # print(self.circles)

        self.lines = []

    def idx_to_coord(self, idx : np.ndarray):
        # idx is the map based numbers, coord is the env based numbers
        x_idx, y_idx = idx

        x_coord = (x_idx * self.res) + self.x_range[0]
        y_coord = ((y_idx * self.res) + self.y_range[0])

        return np.array([x_coord, y_coord])

    def coord_to_idx(self, coord : np.ndarray):
        # idx is the map based numbers, coord is the env based numbers
        x, y = coord
        # print(x, y)
        x_idx = np.where(self.x_points < x)[0][-1]
        y_idx = np.where(self.y_points < y)[0][-1]

        return np.array([x_idx, y_idx])

    def update_map(self, sensor_position, sensor_readings):
        # Sensor Position: Single NumpyState (or Numpy Array) of format (X, Y)
        # Sensor Readings: Set of Lidar Readings for different angles

        x, y = self.coord_to_idx(sensor_position.value)
        self.map[x, y] = EMPTY
        for reading in sensor_readings:
            _, point, _, last_point = reading
            if point:
                idx = self.coord_to_idx(point.value)
                
                # self.map[idx[1], idx[0]] = 1

                # print(line_seg_to_points_dist(sensor_position.value, point.value, self.circles[:, :2]).shape)

                dists = line_seg_to_points_dist(sensor_position.value, point.value, self.circles[:, :2])
                mask = dists < (self.circles[:, 2] * math.sqrt(2))

                inds = self.inds[mask]

                new_mask = (self.map[inds[:, 0], inds[:, 1]] == UNKNOWN)
                inds = inds[new_mask]

                self.map[inds[:, 0], inds[:, 1]] = EMPTY
                self.map[idx[0], idx[1]] = OCCUPIED

                self.lines.append(((x, y), self.coord_to_idx(point.value)))
            else:
                dists = line_seg_to_points_dist(sensor_position.value, last_point.value, self.circles[:, :2])
                mask = dists < self.circles[:, 2]
                inds = self.inds[mask]

                new_mask = (self.map[inds[:, 0], inds[:, 1]] == UNKNOWN)
                inds = inds[new_mask]

                self.map[inds[:, 0], inds[:, 1]] = EMPTY

                # for ind in inds:
                #     if self.map[ind[0], ind[1]] == UNKNOWN:
                #         self.map[ind[0], ind[1]] = EMPTY

                # self.map[inds[:, 0], inds[:, 1]] = EMPTY

    def buffer_obstacles(self, spread_value=0.4):

        kernel = np.array([
                [1,1,1,1,1],
                [1,1,1,1,1],
                [1,1,0,1,1],
                [1,1,1,1,1],
                [1,1,1,1,1]
            ])

        # Convolve: counts neighbors of "1"s
        neighbor_mask = convolve2d((self.map == OCCUPIED).astype(int), kernel, mode="same", boundary="fill", fillvalue=0)

        # Where neighbor_mask > 0 (adjacent to a 1) and current value != 1
        self.map[(neighbor_mask > 0) & (self.map != 1) & (self.map != UNKNOWN)] = spread_value


    def backtrack(self, visited, end):
        path = []
        node = end
        while node:
            path.append(node)
            node = visited[node]

        return path[::-1]
                
    def search(self, start, target):
        start_idx = self.coord_to_idx(start.value)
        target_idx = self.coord_to_idx(target.value)

        q = []
        heappush(q, (0, (start_idx[0], start_idx[1]), None))
        visited = {}

        while q:
            dist, (x, y), parent = heappop(q)

            if target_idx[0] == x and target_idx[1] == y:
                visited[(x,y)] = parent
                print("Found Goal")
                return self.backtrack(visited, tuple(target_idx))

            if (x, y) in visited:
                continue

            visited[(x,y)] = parent

            # neighbors = [(x+1,y),(x-1,y),(x,y-1),(x,y+1),(x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)]
            neighbors = [(x+1,y),(x-1,y),(x,y-1),(x,y+1)]

            for nx, ny in neighbors:
                if (nx >= 0 and nx < len(self.x_idxes) and ny >= 0 and ny < len(self.y_idxes)) and self.map[nx,ny] < 0.51:
                    heappush(q, (dist + self.map[nx,ny], (nx, ny), (x,y)))
        return []

    def add_path_to_map(self, path):
        if path:
            for p in path:
                self.map[p[0], p[1]] = PATH
        
    
    def draw_map(self, ax):
        ax.imshow(np.rot90(self.map, k=1))
        ax.minorticks_on()
        # ax.yaxis.set_inverted(False)

        # ax.invert_yaxis()
        # ax.invert_xaxis()
        # ax.rot90()

        ax.grid(which='minor', linestyle=':', alpha=0.6)
        ax.grid(which='major', linestyle=':', linewidth=0.6)

        # for line in self.lines:
        #     ax.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]])

            # ax.plot([line[0][1], line[1][0]], [line[0][0], line[1][1]])
            # ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]])



if __name__ == '__main__':
    np.random.seed(0)
    om = OccupancyMap()
    os = BiasedPassage(num_walls=1)
    # os = RandomSamplePassage(num_walls=1)
    # os = ParkingSpace()
    env = PointRobot()
    env.set_obstacles(os)
    env.draw_environment(plt.gca())
    plt.show()
    # ls = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, os)
    ls = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, os)

    # readings = ls.read_sensor(np.array((5.0,5.0)))

    sensor_loc = np.array((3.68408613, 2.37189632))

    readings = ls.read_sensor(sensor_loc)
    print(np.array([r[1].value for r in readings if r[1] is not None]))
    # readings = ls.read_sensor(np.array((13.71025438, 7.16023921)))



    # om.draw_map(plt.gca())
    # plt.show()

    # om.update_map(ls.engine.make_state(np.array([5.0, 5.0])), readings)
    om.update_map(ls.engine.make_state(sensor_loc), readings)
    om.draw_map(plt.gca())
    plt.show()
    # exit()
    all_lps = []
    for i in range(20):
        print(f"Running Iteration: {i}")
        loc = ls.engine.sample_valid_point()
        readings = ls.read_sensor(loc)
        om.update_map(loc, readings)

        lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
        all_lps.append(lidar_points)

        # env.draw_environment(plt.gca())
        # plt.title(loc.value)
        # plt.scatter(loc.value[0], loc.value[1], marker='*', color='green')
        # plt.scatter(lidar_points[:, 0], lidar_points[:, 1], color='blue')
        # plt.show()
    om.draw_map(plt.gca())
    plt.show()

    om.buffer_obstacles()
    om.draw_map(plt.gca())
    plt.show()

    env.draw_environment(plt.gca())
    all_lps = np.vstack(all_lps)
    plt.scatter(all_lps[:, 0], all_lps[:, 1], color='blue')
    plt.show()


    start = ls.engine.make_state(np.array([5.0,5.0]))
    target = ls.engine.make_state(np.array([15.0,5.0]))
    # target = ls.engine.make_state(np.array([18.0,1.0]))

    path = om.search(start, target)
    om.add_path_to_map(path)

    om.draw_map(plt.gca())
    plt.show()



    



    # xi, yi = om.coord_to_idx(np.array([7.25, 5.45]))

    # print(xi, yi)
    # print(om.x_points[xi], om.y_points[yi])