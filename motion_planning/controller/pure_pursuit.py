import numpy as np
import math
from sklearn.neighbors import KDTree

from motion_planning.tools import Path
from motion_planning.space import RobotSpace

class PurePursuit():
    def __init__(self, space: RobotSpace, path: Path, dt: float):
        self.space: RobotSpace = space
        self.path: Path = path
        self.dt: float = dt

        self.lookahead_distance = 1.5

        self.path_states = np.stack([state.value for state in self.path], axis=0) # (N, d)
        self.path_edges = np.stack((self.path_states[:-1], self.path_states[1:]), axis=2) # (N, d, 2)
    
    def get_lookahead_point(self, state):
        dists = np.linalg.norm(self.path_states - state, axis=1)
        inside_circle_mask = dists < self.lookahead_distance

        # If the goal is inside the circle, the lookahead point is simply the goal (which is the last point in the path)
        if inside_circle_mask[-1]:
            return self.path_states[-1]
        
        crossing_edges = np.logical_xor(inside_circle_mask[:-1], inside_circle_mask[1:])

        intersecting_edges = self.path_edges[crossing_edges]

        if len(intersecting_edges) == 0:
            # Pick the closest point
            closest_point_idx = np.argmin(dists)
            return self.path_states[closest_point_idx]
        
        intersecting_edge = intersecting_edges[-1]

        # TODO: Get the point that this edge intersects with the circle (But for now we will just do the second point)
        return intersecting_edge[:, 1]

    def get_actions(self, state):
        lookahead_point = self.get_lookahead_point(state)
        d_state = lookahead_point - state

        dist = np.linalg.norm(d_state)

        MAX_VEL = 1
        K_P = 0.8

        if np.all(lookahead_point == self.path_states[-1]):
            velocity = min(MAX_VEL, K_P * dist)
        else:
            velocity = MAX_VEL
        
        dimensional_velocity = velocity * (d_state / dist)
        return dimensional_velocity

def draw_path(ax, env, path, pp, state):
    from matplotlib.collections import LineCollection
    from shapely import Point
    env.draw_environment(ax)
    path_states = np.array([state.value for state in path])
    path = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
    ax.add_collection(LineCollection(path, color='red'))
    # ax.scatter(path_states[:, 0], path_states[:, 1], color='blue')

    circle = Point([state[0], state[1]]).buffer(pp.lookahead_distance)
    ax.plot(*circle.exterior.xy, color='green')

def simulate(env, path, start):
    import matplotlib.pyplot as plt
    from motion_planning.utils import euclidean_distance
    pp = PurePursuit(env, path, 0.1)

    all_states = []

    state = start.value
    while euclidean_distance(state, path[-1].value) > 0.1:
        all_states.append(state)
        simulated_dt = np.random.uniform(0, 0.4)
        actions = pp.get_actions(state)
        state_derivative = actions * simulated_dt
        state = state + state_derivative

        draw_path(plt.gca(), env, path, pp, state)
        env.draw_state(plt.gca(), env.make_state(state))
        all_states_numpy = np.array(all_states)
        plt.plot(all_states_numpy[:, 0], all_states_numpy[:, 1])
        # plt.show()
        plt.pause(0.1)
        plt.cla()

        


    

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt

    from motion_planning.search import RRT, PRM
    from motion_planning.space import DiscRobot, PolygonalRobot, PlanarMobileArm
    from motion_planning.obstacle_sets import BiasedPassage, RandomSamplePassage
    from motion_planning.utils import set_numpy_seed

    save_path = "saves/paths/disc_robot_bpe1_path.pickle"

    # set_numpy_seed(1)

    # env = DiscRobot(disc_radius=0.25)
    # env.set_obstacles(BiasedPassage(num_walls=1))

    # env = PolygonalRobot()
    env = PlanarMobileArm()
    env.set_obstacles(RandomSamplePassage(num_walls=3, gap_width=2))

    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([35.0, 5.0]))
    # start, target = env.make_state(np.array([5.0, 5.0, 0.0])), env.make_state(np.array([35.0, 5.0, 0.0]))
    start, target = env.make_state(np.array([5.0, 5.0, 0.0, 0.0, 0.0])), env.make_state(np.array([35.0, 5.0, 0.0, 0.0, 0.0]))

    # rrt = RRT(env)
    # path = rrt.search(start, target, max_steps=10000, goal_bias=0.1, animate_search_tree=True)
    prm = PRM(env, num_samples=5000, num_neighbors=5)
    prm.create_graph()
    path = prm.search(start, target)
    print(f"Length of Path: {len(path)}")

    # rrt.draw_tree(plt.gca(), path, show_task=True)
    # plt.show()

    
    # pickle.dump(path, open(save_path, 'wb'))

    # path = pickle.load(open(save_path, "rb"))

    pp = PurePursuit(env, path, 0.1)

    draw_path(plt.gca(), env, path, pp, start.value)
    lookahead = pp.get_lookahead_point(start.value)
    plt.scatter(lookahead[0], lookahead[1], color='pink')
    plt.show()

    # print(pp.get_lookahead_point(start.value))

    simulate(env, path, start)






    