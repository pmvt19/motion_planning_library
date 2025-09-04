import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from space import SkidSteerCar, DubinsCar
from matplotlib.collections import LineCollection
from state import NumpyState
from utils import smooth_path, interpolate_edge
from rrt import RRT
from path import KinodynamicPath
import pickle
from obstacle_sets import TestSet, ParkingSpace

class KinodynamicRRT(RRT):
    def __init__(self, env, goal_radius=0.5, max_time_horizon=4, expansion_strategy='single', expansion_attempts=10):
        super().__init__(env=env)
        self.env = env
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        self.goal_radius = goal_radius
        self.max_time_horizon = max_time_horizon
        self.expansion_strategy = expansion_strategy # 'single' or 'sampled_point_bias'
        self.expansion_attempts = expansion_attempts


    def expand_node(self, node, sampled_point):
        if self.env.dist(node.value, self.target.value) < self.goal_radius:
            new_node = self.target
            controls = self.env.make_control(np.zeros(self.env.control_dim))
            time = 0
        else:
            time_horizon = np.random.uniform(low=0, high=self.max_time_horizon) # Choose extension simulation time between [0, self.max_time_horizon)
            new_node, controls, time = self.env.extend_state(node, time_horizon)
            if self.expansion_strategy == 'sampled_point_bias':
                for i in range(self.expansion_attempts-1):
                    time_horizon = np.random.uniform(low=0, high=self.max_time_horizon)
                    potential_new_node, potential_controls, potential_time = self.env.extend_state(node, time_horizon)
                    if self.env.dist(potential_new_node, sampled_point) < self.env.dist(new_node, sampled_point):
                        new_node, controls, time = potential_new_node, potential_controls, potential_time
            

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
    
    def init_search(self, start, target, starting_tree_info=None):
        super().init_search(start, target, starting_tree_info)
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
    # seed = 91
    # seed = 30
    # seed = 81
    # seed = 53
    # seed = 45 # USE FOR SKID STEER CAR AND PARKING SPACE
    # seed = 0 # WORKS FOR DUBINS CAR AND FIXED TASK
    # seed = 60
    # seed = 57
    # seed = 59
    
    print(f"Setting Seed: {seed}") 
    np.random.seed(seed)
    # env = CarParkingEnv()
    # env = DubinsCarEnv()
    env = DubinsCar()
    # env = SkidSteerCar()
    env.set_obstacles(ParkingSpace())
    # start, target = env.sample_task()
    # start, target = env.sample_point(), env.sample_point()
    start, target = env.sample_valid_point(), env.sample_valid_point()
    # start = env.make_state(np.array([3.0, 2.75, 0, 0, 0]))
    # target = env.make_state(np.array([-4.5, -4.75, 0, 0, 0]))

    # start, target = env.get_fixed_task()
    # start = env.make_state(np.array([2.0, 2.75, 0]))
    # target = env.make_state(np.array([-3.0, -2.25, 0]))

    # rrt = KinodynamicRRT(env, goal_radius=1, max_time_horizon=0.1) # For use in CarParkingEnv
    # rrt = KinodynamicRRT(env, goal_radius=4, max_time_horizon=0.5) # For use in Dubins Car
    # rrt = KinodynamicRRT(env, goal_radius=4, max_time_horizon=0.5) # For use in Dubins Car
    rrt = KinodynamicRRT(env, goal_radius=4, max_time_horizon=0.5)

    # state = env.make_state(np.array([6.8738416, 7.59822891, 1.62531756, 0.44829068, 3.03860264]))
    # control = env.make_control(np.array([2.76940055, 0.55987465]))
    # print(env.extend_state(state, 0.4, control)[0].value)
    # exit()
    # path = rrt.search(start, target, max_steps=5000, goal_bias=0.4)
    path = rrt.search(start, target, max_steps=2000, goal_bias=0.1)
    # path = rrt.search(start, target, max_steps=5000, goal_bias=0.1)
    print(len(rrt.tree))
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

