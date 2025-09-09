import numpy as np
import matplotlib.pyplot as plt

from lidar import Lidar
from obstacle_sets import BiasedPassage

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

class OccupancyMap():
    def __init__(self, resolution=0.1):

        self.res = resolution

        # self.x_range = [-10,10]
        # self.y_range = [-10,10]

        self.x_range = [0, 20]
        self.y_range = [0, 10]

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

        x_ind, y_ind = np.meshgrid(self.x_idxes.astype(np.int32), self.y_idxes.astype(np.int32))
        self.inds = np.stack((x_ind, y_ind), axis=2)
        self.inds = self.inds.reshape(-1, 2)

        # print(self.x_idxes)
        # print(self.y_idxes)

        # self.map = np.zeros((len(self.x_points), len(self.y_points)))
        self.map = np.ones((len(self.x_points), len(self.y_points))) * 0.5

        x_cir, y_cir = np.meshgrid(self.x_points_centered, self.y_points_centered)

        self.circles = np.stack((x_cir, y_cir), axis=2)
        self.circles = self.circles.reshape(-1, 2)
        self.circles = np.hstack((self.circles, np.ones((len(self.circles), 1)) * (self.res/2)))

        # print(self.circles.shape)
        # print(self.circles)

    def idx_to_coord(self, idx : np.ndarray):
        # idx is the map based numbers, coord is the env based numbers
        pass

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
        self.map[x, y] = 0
        for reading in sensor_readings:
            _, point, _ = reading
            if point:
                idx = self.coord_to_idx(point.value)
                self.map[idx[0], idx[1]] = 1
                # self.map[idx[1], idx[0]] = 1

                # print(line_seg_to_points_dist(sensor_position.value, point.value, self.circles[:, :2]).shape)

                dists = line_seg_to_points_dist(sensor_position.value, point.value, self.circles[:, :2])
                mask = dists < self.circles[:, 2]

                inds = self.inds[mask]

                self.map[inds[:, 0], inds[:, 1]] = 0

                
        
    
    def draw_map(self, ax):
        ax.imshow(self.map)
        ax.minorticks_on()
        # ax.yaxis.set_inverted(False)
        # ax.rot90()
        ax.grid(which='minor', linestyle=':', alpha=0.6)
        ax.grid(which='major', linestyle=':', linewidth=0.6)

        # mx, my = np.meshgrid(self.x_points, self.y_points)

        # ax.scatter(self.x_points, self.y_points, marker='*', s=1)
        # ax.scatter(mx, my, marker='*', s=1)
        # ax.scatter(self.x_points_centered, self.y_points_centered, marker='^', s=1)
        # ax.scatter(self.cell_points_x, self.cell_points_y, marker='*', s=1)
        # ax.scatter(self.cell_points_centers_x, self.cell_points_centers_y, marker='^', s=1)


if __name__ == '__main__':
    om = OccupancyMap()
    ls = Lidar(0, (0, 2*np.pi), 100, 4.9, BiasedPassage(num_walls=1))

    readings = ls.read_sensor(np.array((5.0,5.0)))



    # om.draw_map(plt.gca())
    # plt.show()

    om.update_map(ls.engine.make_state(np.array([5.0, 5.0])), readings)
    om.draw_map(plt.gca())
    plt.show()

    for i in range(10):
        print(f"Running Iteration: {i}")
        loc = ls.engine.sample_valid_point()
        readings = ls.read_sensor(loc)
        om.update_map(loc, readings)
    om.draw_map(plt.gca())
    plt.show()

    



    # xi, yi = om.coord_to_idx(np.array([7.25, 5.45]))

    # print(xi, yi)
    # print(om.x_points[xi], om.y_points[yi])