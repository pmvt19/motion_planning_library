import time

import matplotlib.pyplot as plt
import numpy as np

from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.space import PointRobot, RobotSpace
from motion_planning.space.approx.circle_approximation import ApproximationSpace


# TODO: Spaces should be able to batch sample points
def temp_batch_sample_points(space: RobotSpace, num_points: int):
    states = []
    for _ in range(num_points):
        states.append(space.sample_point().value)
    states = np.array(states)
    return states


def single_batch_size_run(space: RobotSpace, batch_size: int, sample_sizes: list[int]):
    approx_space = ApproximationSpace(space, batch_size, do_overapproximation=True)

    validation_times = []
    for sample_size in sample_sizes:
        st = time.time()
        batch_sampled_points = temp_batch_sample_points(approx_space, sample_size)
        et = time.time()

        print(f"Sample Size: {sample_size}")

        print(f"Time to Sample Points: {et - st}")

        st = time.time()
        approx_space.batch_is_valid(batch_sampled_points)
        et = time.time()

        validation_time = et - st

        print(f"Time to Validate Points: {validation_time}")
        print("--------------------------------")

        validation_times.append(validation_time)

    return validation_times


if __name__ == "__main__":
    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=3))

    sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000, 130000]
    batch_sizes = [100, 1000, 5000, 10000, 15000, 20000, 35000, 100000]

    avg_validation_times = []

    ## Overall Validation Time Plot
    for batch_size in batch_sizes:
        print(f"Doing Batch Size: {batch_size}")
        validation_times = single_batch_size_run(env, batch_size, sample_sizes)
        avg_validation_times.append(np.sum(validation_times) / np.sum(sample_sizes))

        plt.plot(sample_sizes, validation_times, label=f"BatchSize: {batch_size}")

    plt.xlabel("Number of Samples")
    plt.ylabel("Validation Time")
    plt.legend()
    plt.show()

    ## Time Per Sample Plot

    # Generate a unique color for each bar index
    # We slice the colormap dynamically based on the total number of categories
    cmap = plt.colormaps['viridis']
    bar_colors = [cmap(i / len(batch_sizes)) for i in range(len(batch_sizes))]

    plt.bar([str(batch_size) for batch_size in batch_sizes], avg_validation_times, color=bar_colors)
    plt.xlabel("Batch Size")
    plt.ylabel("Validation Time Per Sample")
    plt.show()
