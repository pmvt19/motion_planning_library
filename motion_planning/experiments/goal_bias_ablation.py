import time

import matplotlib.pyplot as plt
import numpy as np

from motion_planning.search import RRT, BiDirectionalRRT
from motion_planning.tools import NumpyState


def repeatedly_test_search(
    search_alg: RRT,
    num_runs: int,
    goal_bias: float,
    start: NumpyState,
    target: NumpyState,
):
    time_tracker = []
    for _ in range(num_runs):
        st = time.time()
        _ = search_alg.search(start, target, goal_bias=goal_bias)
        et = time.time()
        time_tracker.append(et - st)
    return time_tracker


def grid_search_goal_bias(
    search_alg: RRT,
    num_runs: int,
    goal_biases: list[float],
    start: NumpyState,
    target: NumpyState,
):
    res_dict = {}
    for goal_bias in goal_biases:
        times = repeatedly_test_search(search_alg, num_runs, goal_bias, start, target)
        res_dict[goal_bias] = times
    return res_dict


def plot_results_dict(results: dict[float, list[float]]):
    means = [np.mean(times) for times in results.values()]
    print(results.keys(), means)
    plt.plot(results.keys(), means)
    plt.show()


if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=3))

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([35.0, 5.0]))

    results = grid_search_goal_bias(
        BiDirectionalRRT(env), 2, [0.0, 0.01, 0.05, 0.1, 0.2], start, target
    )

    plot_results_dict(results)
