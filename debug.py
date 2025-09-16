from space import PointRobot
from rrt import RRT
import numpy as np 
import time
import matplotlib.pyplot as plt
from state import NumpyState, AngularNumpyState
from circle_approximation import ApproximationSpace
from obstacle_sets import ParkingSpace

from utils import batch_interpolate_edge, batch_interpolate_edge_uniform


if __name__ == '__main__':
    np.random.seed(0)

    # numpystate = NumpyState(value=np.array([0.0, 0.0]))
    # angularstate = AngularNumpyState(value=np.array([0.0, 0.0, 0.0]), angular_dims_start=2)

    # print(isinstance(numpystate, NumpyState), isinstance(numpystate, AngularNumpyState), isinstance(angularstate, NumpyState), isinstance(angularstate, AngularNumpyState))
    # exit()
    # env1 = OpenSpace2d()
    env1 = PointRobot()
    env2 = PointRobot()

    start1, target1 = env1.sample_task()

    start2, target2 = env2.sample_task()

    print("Task 1:", start1.value, target1.value)
    print("Task 2:", start2.value, target2.value)

    starts = np.array([
        start1.value,
        start2.value
    ])

    targets = np.array([
        target1.value,
        target2.value
    ])

    print(batch_interpolate_edge(starts, targets, 0.5, None))
    print(batch_interpolate_edge_uniform(starts, targets, 0.5, None))



    # # env1.is_valid_edge(start, target)
    # # env2.is_valid_edge(start, target)
    # # env1.shoot_ray(start, target, delta=0.5)
    # # env2.shoot_ray(start, target, delta=0.5)

    # goal_bias = 0.05

    # start, target = env1.sample_task()
    # print(start.value, target.value)
    # rrt = RRT(env=env1)
    # path = rrt.search(start, target, max_steps=1000, goal_bias=goal_bias, animate_search_tree=False)
    # rrt.draw_tree(plt.gca(), path=path)
    # plt.show()

    # rrt = RRT(env=env2)
    # path = rrt.search(start, target, max_steps=1000, goal_bias=goal_bias, animate_search_tree=False)
    # rrt.draw_tree(plt.gca(), path=path)
    # plt.show()

    # env2.animate_path(path, frame_delay=0.1)


    # old_method_time = []
    # new_method_time = []

    # for i in range(10000):
    #     start, target = env1.sample_task()

    #     start_time = time.time()
    #     # env1.is_valid_edge(start, target)
    #     env1.shoot_ray(start, target, delta=0.5)
    #     end_time = time.time()
    #     old_method_time.append(end_time-start_time)

    #     start_time = time.time()
    #     # env2.is_valid_edge(start, target)
    #     env2.shoot_ray(start, target, delta=0.5)
    #     end_time = time.time()
    #     new_method_time.append(end_time-start_time)

    # print(f"Old Method Average: {np.mean(old_method_time)}")
    # print(f"New Method Average: {np.mean(new_method_time)}")

    # env = PointRobot()
    # env.set_obstacles(ParkingSpace())
    # env = ApproximationSpace(env, batch_size=1000, do_overapproximation=False)

    # start_states = np.array([[0.5433628, 2.26368434]])
    # end_states = np.array([[-0.59120828, 1.67877715]])

    # # validities, pts = env.batch_is_valid_edge_debug(start_states, end_states)
    # # validities, pts = env.batch_is_valid_edge_debug(end_states, start_states)

    # env.draw_environment(plt.gca())
    # for pt in pts:
    #     plt.scatter(pt[0], pt[1])
    # plt.show()


        