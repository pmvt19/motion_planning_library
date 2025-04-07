import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from environments import CarParkingEnv, DubinsCarEnv
from matplotlib.collections import LineCollection
from state import NumpyState
from utils import smooth_path, interpolate_edge
from rrt import RRT
from path import KinodynamicPath
import pickle

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
    
    def backtrack(self, end):
        if end not in self.child_to_parent:
            return KinodynamicPath()
        
        path = []
        control_seq = []
        node = end 
        control = None
        while node:
            path.append(node)
            parent_info = self.child_to_parent[node]
            node = parent_info[0]
            control = (parent_info[1], parent_info[2])
            control_seq.append(control)

        return KinodynamicPath(path=path[::-1][:-1], controls=control_seq[::-1][1:-1], dt=self.env.dt)
    
    def init_search(self, start, target):
        super().init_search(start, target)
        self.tree[self.start] = []
        self.child_to_parent[self.start] = (None, None, None)

if __name__ == "__main__":
    seed = np.random.randint(0, 100) # Use seed 6 for an interesting path
    # seed = 15
    # Dubin's Car Environment
    # seed = 49 
    # seed = 9
    # seed = 4 #debug
    # seed = 1
    # seed = 42
    # seed = 21
    # seed = 85
    # seed = 86 # Interesting S-shaped Path
    print(f"Setting Seed: {seed}") 
    np.random.seed(seed)
    # env = CarParkingEnv()
    env = DubinsCarEnv()
    start, target = env.sample_task()
    # start, target = env.get_fixed_task()
    # start = env.make_state(np.array([2.0, 2.75, 0]))
    # target = env.make_state(np.array([-3.0, -2.25, 0]))

    # rrt = KinodynamicRRT(env, goal_radius=1, max_time_horizon=0.1) # For use in CarParkingEnv
    # rrt = KinodynamicRRT(env, goal_radius=4, max_time_horizon=0.5) # For use in Dubins Car
    rrt = KinodynamicRRT(env, goal_radius=4, max_time_horizon=0.5) # For use in Dubins Car

    # state = env.make_state(np.array([6.8738416, 7.59822891, 1.62531756, 0.44829068, 3.03860264]))
    # control = env.make_control(np.array([2.76940055, 0.55987465]))
    # print(env.extend_state(state, 0.4, control)[0].value)
    # exit()
    path = rrt.search(start, target, max_steps=5000, goal_bias=0.4)
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()

    # env.animate_path(path, frame_delay=0.001)

    # rrt.draw_tree(plt.gca(), path=path, hold=True)

    controls = path.controls
    state_seqs = env.simulate(start, controls)

    env.animate_path(state_seqs, frame_delay=0.1)

    # for p in path:
        # print(p.value)

    # for i in range(len(path)):
    #     print(path[i].value, controls[i][0].value, controls[i][1])

