import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from environments import Environment2d, RandomSamplePassage, OpenSpace2d, CarParkingEnv
from matplotlib.collections import LineCollection
from state import NumpyState
from utils import smooth_path
from path import Path

class RRT():
    def __init__(self, env, delta=0.5):
        self.env = env
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        self.delta = delta

    def select_node(self, goal_bias=0):
        if np.random.random() < goal_bias:
            sampled_point = self.target
        else:
            sampled_point = self.env.sample_valid_point()
        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point

    def expand_node(self, node, sampled_point):

        if np.linalg.norm(self.target.value - node.value) < self.delta:
            new_node = self.target
        else:
            new_node = self.env.shoot_ray(node, sampled_point, self.delta)

        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node

        return new_node

    def draw_tree(self, ax, path:Path = None, show_task=True):
        self.env.draw_environment(ax)
        nodes = np.array([node.value for node in self.tree.keys()])
        try:
            ax.scatter(nodes[:, 0], nodes[:, 1])
            if show_task:
                ax.scatter(self.start.value[0], self.start.value[1], s=100, c='green')
                ax.scatter(self.target.value[0], self.target.value[1], s=100, c='red')

            edges = [[(p.value[0], p.value[1]), (c.value[0], c.value[1])] for p in self.tree for c in self.tree[p]]
            edges = LineCollection(edges, color='blue')
            ax.add_collection(edges)

        except Exception as e:
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            print(f"{FAIL}Cannot Draw Tree Without Running Search{ENDC}")
            raise Exception

        if path:
            path = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
            ax.add_collection(LineCollection(path, color='red'))

    def draw_voronoi_diagram(self):
        nodes = np.array([node.value for node in self.tree.keys()])
        vor = Voronoi(nodes)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
        self.draw_tree(fig.gca(), hold=True)

    # def backtrack(self):
    #     if self.target not in self.child_to_parent:
    #         return Path()

    #     path = []
    #     node = self.target 
    #     while node:
    #         path.append(node)
    #         node = self.child_to_parent[node]
    #     return Path(path=path[::-1])

    def backtrack(self, end=None):
        if end is None or end not in self.child_to_parent:
            return Path()

        path = []
        node = end
        while node:
            path.append(node)
            node = self.child_to_parent[node]
        return Path(path=path[::-1])
    
    def init_search(self, start, target):
        self.start = start 
        self.target = target
        self.tree[self.start] = []
        self.child_to_parent[self.start] = None

    def search(self, start, target, max_steps=10000, goal_bias=0.1, animate_search_tree=False):
        self.init_search(start, target)

        cur_node = start
        num_steps = 0

        start_time = time.time()
        while (cur_node != target and num_steps < max_steps):
            print(f"Searching Step: {num_steps}", end='\r')
            exp_node, sampled_point = self.select_node(goal_bias=goal_bias)
            cur_node = self.expand_node(exp_node, sampled_point)
            num_steps += 1
            if animate_search_tree:
                self.draw_tree(plt.gca())
        search_time = time.time() - start_time
        print(f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}")
        path = self.backtrack(end=target)
        
        return path

class BiDirectionalRRT():
    def __init__(self, env, delta=0.5, max_connection_distance=1):
        self.env = env
        self.delta = delta
        self.max_connection_distance = max_connection_distance

    def rrt_step(self, rrt, goal_bias):
        exp_node, sampled_point = rrt.select_node(goal_bias=goal_bias)
        cur_node = rrt.expand_node(exp_node, sampled_point)
        return cur_node

    def attempt_tree_connection(self, forward_rrt, backward_rrt):
        forward_tree_nodes = np.array([node.value for node in forward_rrt.tree.keys()])
        backward_tree_nodes = np.array([node.value for node in backward_rrt.tree.keys()])

        kdt = KDTree(forward_tree_nodes)
        dist, ind = kdt.query(backward_tree_nodes, k=1)
        dist = dist.flatten()
        ind = ind.flatten()

        if np.min(dist) < self.max_connection_distance:
            backward_tree_node_idx = np.argmin(dist)
            forward_tree_node_idx = ind[backward_tree_node_idx]
            return self.env.is_valid_edge(forward_tree_nodes[forward_tree_node_idx], backward_tree_nodes[backward_tree_node_idx]), \
                    (self.env.make_state(forward_tree_nodes[forward_tree_node_idx]), self.env.make_state(backward_tree_nodes[backward_tree_node_idx]))
        return False, None
    
    def backtrack(self, forward_rrt, backward_rrt, connection):
        if connection is None:
            return Path()
        forward_end_state, backward_end_state = connection
        forward_path = forward_rrt.backtrack(end=forward_end_state)
        backward_path = backward_rrt.backtrack(end=backward_end_state)

        # Join Both Paths and Reverse the backward RRT Tree path
        joined_path = forward_path.path + backward_path.path[::-1] 
        return Path(path=joined_path)

    def search(self, start, target, max_steps=10000, goal_bias=0.1):
        self.forward_rrt = RRT(env=self.env, delta=self.delta)
        self.backward_rrt = RRT(env=self.env, delta=self.delta)

        self.forward_rrt.init_search(start, target)
        self.backward_rrt.init_search(target, start)

        num_steps = 0
        is_connected = False
        start_time = time.time()

        while not is_connected and num_steps < max_steps:
            print(f'Searching Step: {num_steps}', end='\r')
            self.rrt_step(self.forward_rrt, goal_bias)
            self.rrt_step(self.backward_rrt, goal_bias)
            is_connected, connection = self.attempt_tree_connection(self.forward_rrt, self.backward_rrt)
            num_steps += 1
        
        search_time = time.time() - start_time
        print(f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}")
        
        path = self.backtrack(forward_rrt=self.forward_rrt, backward_rrt=self.backward_rrt, connection=connection)
        return path
        
    def draw_tree(self, ax, path : Path = None, hold=False, show_task=True):
        self.env.draw_environment(ax)
        self.backward_rrt.draw_tree(ax, show_task=False)
        self.forward_rrt.draw_tree(ax, path=path, show_task=show_task)

if __name__ == "__main__":
    seed = np.random.randint(0, 100)
    seed = 22
    print(f"Setting Seed: {seed}")
    
    np.random.seed(seed)
    start = (0, 0)
    target = (9, 9)
    
    env = OpenSpace2d()
    # env = Environment2d()
    # env = RandomSamplePassage()
    # env = CarParkingEnv()
    # start, target = env.sample_task()

    # start = env.make_state(np.array([0,0]))
    start = env.make_state(np.array([-9,-9]))
    target = env.make_state(np.array([9,9]))

    max_steps = 10
    goal_bias = 0.1

    rrt = RRT(env)
    # rrt = BiDirectionalRRT(env)
    path = rrt.search(start, target, max_steps=10000, goal_bias=0.1)
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()
    
    # smoothed_path = smooth_path(env, path)
    # rrt.draw_tree(plt.gca(), path=smoothed_path)

    # from kinematic_path_smoothing import smooth_path_trajectory_optimization
    # smoothed_path = smooth_path_trajectory_optimization(env, path)

    # rrt.draw_voronoi_diagram()
