import numpy as np
from utils import interpolate_edge, batch_interpolate_edge, batch_interpolate_edge_uniform
from space import PointRobot, PolygonalRobot, PlanarMobileArm

if __name__ == '__main__':
    seed = np.random.randint(low=0, high=100)
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)
    # env = PolygonalRobot()
    # env = PointRobot()
    # env = PlanarMobileArm()

    # N = 2
    # starts = np.array([env.sample_valid_point().value for _ in range(N)])
    # ends = np.array([env.sample_valid_point().value for _ in range(N)])

    # print("Starts")
    # print(starts)
    # print("ends")
    # print(ends)

    # print("pts and gradients")
    # pts, steps = batch_interpolate_edge(starts, ends, env.edge_validity_delta, env.angular_dims_start)
    # print(pts, steps)
    # print(pts.shape, steps.shape)

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


    # start, end = env.sample_valid_point(), env.sample_valid_point()

    # start = env.make_state(np.array([-8.38802066, 2.61335129, 0.07028609]))
    # end = env.make_state(np.array([6.28713603, 2.07742703, 3.58198959]))

    # # start = env.make_state(np.array([0, 0, 0.1]))
    # # end = env.make_state(np.array([0, 0, 1.1]))

    # # start = env.make_state(np.array([0, 0, np.pi/6]))
    # # end = env.make_state(np.array([0, 0, 11*np.pi/6]))

    # # start = env.make_state(np.array([0, 0, np.pi/3]))
    # # end = env.make_state(np.array([0, 0, 2*np.pi/3]))

    # # start = env.make_state(np.array([0, 0, 11*np.pi/6]))
    # # end = env.make_state(np.array([0, 0, 2*np.pi/3]))

    # # start = env.make_state(np.array([0, 0, 7*np.pi/4]))
    # # end = env.make_state(np.array([0, 0, np.pi/4]))
    # edge_states = env.get_edge_states(start.value, end.value)
    # print(edge_states)

    # print(interpolate_edge(start, end, env.edge_validity_delta))

    # print("Gradient Literal")
    # print(env.make_state(env.make_state(end.value - start.value).value + start.value).value)
    # print(np.pi/4)

    # print(-np.pi/6 + 1.49665859, 2*np.pi/3)

    # print("Desired Gradient:", 2*np.pi/3-(-np.pi/6))

    # print(np.pi/6*2)
    