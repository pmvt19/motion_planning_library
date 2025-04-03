import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from environments import CarParkingEnv
from matplotlib.collections import LineCollection
from state import NumpyState
from utils import smooth_path, interpolate_edge
from rrt import RRT

class KinodynamicRRT(RRT):
    def __init__(self, env, delta=0.5, max_time_horizon=4):
        super().__init__(env=env)
        self.env = env
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        self.delta = delta
        self.max_time_horizon = max_time_horizon

    # def select_node(self, goal_bias=0):
    #     if np.random.random() < goal_bias:
    #         sampled_point = self.target
    #     else:
    #         sampled_point = self.env.sample_valid_point()
    #     nodes = np.array([node.value for node in self.tree.keys()])
    #     kdt = KDTree(nodes)
    #     _, ind = kdt.query(np.array([sampled_point.value]), k=1)
    #     idx = ind[0][0]
    #     return self.env.make_state(nodes[idx]), sampled_point

    def expand_node(self, node, sampled_point):

        if self.env.dist(node.value, self.target.value) < self.delta:
            new_node = self.target
        else:
            time_horizon = np.random.random() * self.max_time_horizon # Choose extension simulation time between [0, self.max_time_horizon)
            new_node, controls = self.env.extend_state(node, time_horizon)

        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node

        return new_node

if __name__ == "__main__":
    seed = np.random.randint(0, 100) # Use seed 6 for an interesting path
    seed = 15
    print(f"Setting Seed: {seed}") 
    np.random.seed(seed)
    env = CarParkingEnv()
    # start, target = env.sample_task()
    start, target = env.get_fixed_task()
    # start = env.make_state(np.array([2.0, 2.75, 0]))
    # target = env.make_state(np.array([-3.0, -2.25, 0]))

    rrt = KinodynamicRRT(env, delta=1, max_time_horizon=0.1)
    path = rrt.search(start, target, max_steps=15000, goal_bias=0.4)
    rrt.draw_tree(plt.gca(), path=path, hold=True)

    env.animate_path(path, frame_delay=0.01)

    # interpolated_path = [env.make_state(state) for edge in [interpolate_edge(path[i].value, path[i+1].value, delta=0.1) for i in range(len(path)-1)] for state in edge]
    # env.animate_path(interpolated_path, frame_delay=0.05)


