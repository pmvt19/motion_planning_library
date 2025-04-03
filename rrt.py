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
            # new_node = tuple(new_node)

        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node

        return new_node

    def draw_tree(self, ax, path=None, hold=False):
        self.env.draw_environment(ax)
        nodes = np.array([node.value for node in self.tree.keys()])
        try:
            ax.scatter(nodes[:, 0], nodes[:, 1])
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
        
        if hold:
            plt.show()
        plt.pause(0.01)
        plt.clf()

    def draw_voronoi_diagram(self):
        nodes = np.array([node for node in self.tree.keys()])
        vor = Voronoi(nodes)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
        self.draw_tree(fig.gca(), hold=True)

    def backtrack(self):
        path = []
        node = self.target 
        while node:
            path.append(node)
            node = self.child_to_parent[node]
        return path[::-1]
        

    def search(self, start, target, max_steps=10000, goal_bias=0.1):
        self.tree[start] = []
        self.child_to_parent[start] = None
        cur_node = start
        num_steps = 0
        
        self.start = start 
        self.target = target

        start_time = time.time()
        while (cur_node != target and num_steps < max_steps):
            print(f"Searching Step: {num_steps}", end='\r')
            exp_node, sampled_point = self.select_node(goal_bias=goal_bias)
            cur_node = self.expand_node(exp_node, sampled_point)
            num_steps += 1
            # self.draw_tree(plt.gca())
        search_time = time.time() - start_time
        print(f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}")
        # draw_tree(env, tree, start, target, hold=True)
        path = None
        if cur_node == target:
            path = self.backtrack()
        
        return path

if __name__ == "__main__":
    seed = np.random.randint(0, 100)
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)
    # start = (0, 0)
    # target = (9, 9)
    
    # env = OpenSpace2d()
    # env = Environment2d()
    # env = RandomSamplePassage()
    env = CarParkingEnv()
    # start, target = env.sample_task()

    start = env.make_state(np.array([0,0]))
    target = env.make_state(np.array([9,9]))

    max_steps = 10
    goal_bias = 0.1

    rrt = RRT(env)
    path = rrt.search(start, target, goal_bias=0.01)
    rrt.draw_tree(plt.gca(), path=path, hold=True)
    
    smoothed_path = smooth_path(env, path)
    rrt.draw_tree(plt.gca(), path=smoothed_path, hold=True)

    # from kinematic_path_smoothing import smooth_path_trajectory_optimization
    # smoothed_path = smooth_path_trajectory_optimization(env, path)

    # rrt.draw_voronoi_diagram()
