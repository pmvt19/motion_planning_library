import numpy as np
from obstacle_sets import BiasedPassage
from lidar import Lidar, SuperOptimizedLidar
from space import PointRobot
import time

import matplotlib.pyplot as plt

def plot_readings(ax, readings, loc, title=''):
    ax.set_title(title)
    lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
    print(f"Num Points for {title}: {len(lidar_points)}")
    ax.scatter(lidar_points[:, 0], lidar_points[:, 1], color='red', zorder=2)
    ax.scatter(loc[0], loc[1], color='green', marker='*')

def plot_lines(ax, loc, farthest_points):
    print(farthest_points.shape)
    for point in farthest_points:
        print(point)
        ax.plot([loc[0], point[0]], [loc[1], point[1]], color='blue')

def plot_bulk_readings(ax, lidar_points, locs, title=''):
    ax.set_title(title)
    print(f"Num Points for {title}: {len(lidar_points)}")
    print(lidar_points.shape)
    ax.scatter(lidar_points[:, 0], lidar_points[:, 1], color='red', zorder=2)
    ax.scatter(locs[:, 0], locs[:, 1], color='green', marker='*')


def do_time_test(op, unop, num_iters=10):
    locs = [lidar_optimized.engine.sample_valid_point().value for i in range(num_iters)]
    st = time.time()
    r_op = [op.read_sensor(locs[i]) for i in range(num_iters)]
    et = time.time()
    print(f"Time to Read {num_iters} Sensor Readings from Optimized: {et-st}")

    st = time.time()
    r_unop = [unop.read_sensor(locs[i]) for i in range(num_iters)]
    et = time.time()
    print(f"Time to Read {num_iters} Sensor Readings from Unoptimized: {et-st}")


if __name__ == "__main__":
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    env.set_obstacles(os)

    num_angles = 360
    max_dist = 4.9

    lidar_optimized = SuperOptimizedLidar(None, (0, 2*np.pi), num_angles, max_dist, os)
    lidar_unoptimized = Lidar(None, (0, 2*np.pi), num_angles, max_dist, os)

    do_time_test(lidar_optimized, lidar_unoptimized)
    exit()

    # loc = np.array([5.0, 5.0])
    loc = np.array([7.5, 2.5])

    readings_optimized = lidar_optimized.read_sensor(loc)
    readings_unoptimized = lidar_unoptimized.read_sensor(loc)

    fig, ax = plt.subplots(2)
    plot_readings(ax[0], readings_optimized, loc, 'Optimized')
    plot_readings(ax[1], readings_unoptimized, loc, 'Unoptimized')

    env.draw_environment(ax[0])
    env.draw_environment(ax[1])

    farthest_points = lidar_optimized.get_farthest_points(loc)
    plot_lines(ax[0], loc, farthest_points)



    plt.show()

    locs = [lidar_optimized.engine.sample_valid_point().value for i in range(10)]
    all_lidar_points_optimized = []
    all_lidar_points_unoptimized = []

    print("Before loops")
    for i in range(10):
        print(f"Running Iteration: {i}")
        loc = locs[i]

        readings = lidar_optimized.read_sensor(loc)
        lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
        all_lidar_points_optimized.append(lidar_points.reshape(-1, 2))

        readings = lidar_unoptimized.read_sensor(loc)
        lidar_points = np.array([r[1].value for r in readings if r[1] is not None])
        all_lidar_points_unoptimized.append(lidar_points.reshape(-1, 2))
    

    all_lidar_points_optimized = np.vstack(all_lidar_points_optimized)
    all_lidar_points_unoptimized = np.vstack(all_lidar_points_unoptimized)
    locs = np.array(locs)

    fig, ax = plt.subplots(2)
    
    plot_bulk_readings(ax[0], all_lidar_points_optimized, locs, 'Optimized')
    plot_bulk_readings(ax[1], all_lidar_points_unoptimized, locs, 'Unoptimized')

    env.draw_environment(ax[0])
    env.draw_environment(ax[1])

    plt.show()
