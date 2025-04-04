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
from path import KinodynamicPath

class KinodynamicRRT(RRT):
    def __init__(self, env, goal_radius=0.5, max_time_horizon=4):
        super().__init__(env=env)
        self.env = env
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        self.goal_radius = goal_radius
        self.max_time_horizon = max_time_horizon

    def expand_node(self, node, sampled_point):

        if self.env.dist(node.value, self.target.value) < self.goal_radius:
            new_node = self.target
            controls = self.env.make_control(np.zeros(self.env.control_dim))
            time = 0
        else:
            time_horizon = np.random.random() * self.max_time_horizon # Choose extension simulation time between [0, self.max_time_horizon)
            new_node, controls, time = self.env.extend_state(node, time_horizon)

        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = (node, controls, time)

        return new_node
    
    def backtrack(self):
        if self.target not in self.child_to_parent:
            return KinodynamicPath()
        
        path = []
        control_seq = []
        node = self.target 
        control = None
        while node:
            path.append(node)
            parent_info = self.child_to_parent[node]
            node = parent_info[0]
            control = (parent_info[1], parent_info[2])
            control_seq.append(control)

        return KinodynamicPath(path=path[::-1][:-1], controls=control_seq[::-1][1:-1], dt=self.env.dt)
    
    def init_search(self):
        self.tree[start] = []
        self.child_to_parent[start] = (None, None, None)

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

    rrt = KinodynamicRRT(env, goal_radius=1, max_time_horizon=0.1)
    path = rrt.search(start, target, max_steps=15000, goal_bias=0.4)
    # rrt.draw_tree(plt.gca(), path=path, hold=True)

    # env.animate_path(path, frame_delay=0.001)

    plt.clf()
    # rrt.draw_tree(plt.gca(), path=path, hold=True)

    controls = path.controls
    state_seqs = env.simulate(start, controls)

    # env.animate_path(state_seqs, frame_delay=0.001)

    print(path.path)
    print(path.controls)
    print(path.dt)


