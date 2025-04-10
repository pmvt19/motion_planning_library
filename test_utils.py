import numpy as np
from utils import interpolate_edge
from space import PointRobot, PolygonalRobot, PlanarMobileArm

if __name__ == '__main__':
    # np.random.seed(0)
    env = PolygonalRobot()
    # start, end = env.sample_valid_point(), env.sample_valid_point()

    start = env.make_state(np.array([-8.38802066, 2.61335129, 0.07028609]))
    end = env.make_state(np.array([6.28713603, 2.07742703, 3.58198959]))

    # start = env.make_state(np.array([0, 0, 0.1]))
    # end = env.make_state(np.array([0, 0, 1.1]))

    # start = env.make_state(np.array([0, 0, np.pi/6]))
    # end = env.make_state(np.array([0, 0, 11*np.pi/6]))

    # start = env.make_state(np.array([0, 0, np.pi/3]))
    # end = env.make_state(np.array([0, 0, 2*np.pi/3]))

    # start = env.make_state(np.array([0, 0, 11*np.pi/6]))
    # end = env.make_state(np.array([0, 0, 2*np.pi/3]))

    # start = env.make_state(np.array([0, 0, 7*np.pi/4]))
    # end = env.make_state(np.array([0, 0, np.pi/4]))
    edge_states = env.get_edge_states(start.value, end.value)
    print(edge_states)

    print(interpolate_edge(start, end, env.edge_validity_delta))

    print("Gradient Literal")
    print(env.make_state(env.make_state(end.value - start.value).value + start.value).value)
    print(np.pi/4)

    print(-np.pi/6 + 1.49665859, 2*np.pi/3)

    print("Desired Gradient:", 2*np.pi/3-(-np.pi/6))

    print(np.pi/6*2)
    