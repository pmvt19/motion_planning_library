import numpy as np
from space import FixedArm
from circle_approximation import ApproximationSpace
from obstacle_sets import TestSet, ParkingSpace
import matplotlib.pyplot as plt
import time
from utils import smooth_path, interpolate_path
import matplotlib

from rrt import RRT

# Interactive element 
# - Shows workspace and cspace states as one uses the controller to manipulate the arm
# - Potentially allow the user to change properties of the robot like arm lengths

# Search element
# - Shows workspace and cspace executing a planned rrt path (Done)
# - Add Colors to CSpace side that shows the progression of the path (Done)

def animate_path_and_space(path, obstacle_points, show_prev=True, frame_delay=0.1):
    cmap = matplotlib.colormaps['viridis']
    colors = [cmap(i/len(path)) for i in range(len(path))]

    fig, axs = plt.subplots(1,2)
    axs[1].scatter(obstacle_points[:, 0], obstacle_points[:, 1], color='red')

    for i, c in enumerate(path.path):
        axs[0].cla()

        if not show_prev:
            axs[1].cla()
            axs[1].scatter(obstacle_points[:, 0], obstacle_points[:, 1], color='red')

        env.draw_environment(axs[0])
        env.draw_state(axs[0], c)
        axs[1].scatter(c.value[0], c.value[1], color=colors[i], marker='^')
        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')
        plt.pause(frame_delay)


if __name__ == '__main__':
    # np.random.seed(0)
    env = FixedArm()
    env.arm_link_lengths = np.array([3,3]) # HACK: DO NOT CHANGE ARM LENGTHS LIKE THIS
    env.set_obstacles(TestSet())
    # env.set_obstacles(ParkingSpace())

    start, target = env.sample_valid_point(), env.sample_valid_point()
    rrt = RRT(env)
    path = rrt.search(start, target, max_steps=1000)

    # env = ApproximationSpace(env)
    start_time = time.time()
    points = np.array([env.sample_point().value for _ in range(10000)])
    end_time = time.time()
    print(f"Time to Sample Points: {end_time-start_time}")

    start_time = time.time()
    point_validities = env.batch_is_valid(points)
    points = points[(point_validities == False)]
    end_time = time.time()
    print(f"Time to Validate Points: {end_time-start_time}")

    # path = smooth_path(env, path)
    path = interpolate_path(path, env, 0.05)
    animate_path_and_space(path, points)

