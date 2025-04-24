import numpy as np
from space import FixedArm
from circle_approximation import ApproximationSpace
from obstacle_sets import TestSet
import matplotlib.pyplot as plt
import time
from utils import smooth_path, interpolate_path


from rrt import RRT

if __name__ == '__main__':
    np.random.seed(0)
    env = FixedArm()
    env.arm_link_lengths = np.array([3,3])
    env.set_obstacles(TestSet())

    start, target = env.sample_valid_point(), env.sample_valid_point()
    rrt = RRT(env)
    path = rrt.search(start, target, max_steps=1000)
    # print(path.path)

    

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

    # plt.scatter(points[:, 0], points[:, 1], color='red')
    # plt.show()

    # plt.clf()
    # env.draw_environment(plt.gca())
    # # print(points[0])
    # print(points.shape)
    # env.draw_state(plt.gca(), env.make_state(points[0]))
    # plt.show()

    # fig, axs = plt.subplots(1,2)
    # env.draw_environment(axs[0])
    # axs[1].scatter(points[:, 0], points[:, 1], color='red')

    # path_configs = np.array([c.value for c in path.path])
    # axs[1].plot(path_configs[:, 0], path_configs[:, 1], color='blue', marker='^')
    # plt.show()

    # env.animate_path(path, frame_delay=0.5)

    # path = smooth_path(env, path)
    # path = interpolate_path(path, env, 0.1)
    # fig, axs = plt.subplots(1,2)
    # for c in path.path:
    #     # plt.clf()
    #     plt.close()
    #     fig, axs = plt.subplots(1,2)
    #     env.draw_environment(axs[0])
    #     env.draw_state(axs[0], c)
    #     # axs[1].scatter(points[:, 0], points[:, 1], color='red')
    #     env.draw_environment(axs[1])
    #     axs[1].scatter(c.value[0], c.value[1], color='blue', marker='^')
    #     plt.pause(0.5)
    #     # time.sleep(0.1)
    #     # plt.close()

    for c in path.path:
        plt.clf()
        # plt.close()
        # fig, axs = plt.subplots(1,2)
        # env.draw_environment(axs[0])
        # env.draw_state(axs[0], c)
        plt.scatter(points[:, 0], points[:, 1], color='red')
        # env.draw_environment(axs[1])
        plt.scatter(c.value[0], c.value[1], color='blue', marker='^')
        plt.pause(0.5)
        # time.sleep(0.1)
        # plt.close()