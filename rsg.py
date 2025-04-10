import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
import time
from matplotlib.collections import LineCollection
from state import NumpyState
from utils import smooth_path
from rrt import RRT 

class RandomSampleGeneration(RRT):
    def __init__(self, env, num_neighbors=5, delta=0.5):
        super().__init__(env, delta=delta)
        self.env = env
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        self.num_neighbors = num_neighbors
        self.delta = delta

    def expand_node(self, node, sampled_point):

        if np.linalg.norm(self.target.value - node.value) < self.delta:
            new_node = self.target
        else:
            noise = np.random.normal(size=(self.num_neighbors, node.value.shape[0]))
            noise = (noise / np.linalg.norm(noise, axis=1, keepdims=True)) * self.delta
            new_node_candidates = node.value + noise
            
            dists = [self.env.dist(new_node_candidates[i], sampled_point.value) if (self.env.is_valid(new_node_candidates[i]) and self.env.is_valid_edge(node.value, new_node_candidates[i])) else float('inf') for i in range(self.num_neighbors)]
            
            if min(dists) == float('inf'):
                new_node = node
            else:
                new_node = new_node_candidates[np.argmin(dists)]
                new_node = self.env.make_state(new_node)

        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node
        return new_node

if __name__ == "__main__":
    # seed = np.random.randint(0, 100)
    seed = 54
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)
    # start = (0, 0)
    # target = (9, 9)
    
    # env = Environment2d()
    env = RandomSamplePassage()
    start, target = env.sample_task()

    # start = env.make_state(np.array([0,0]))
    # target = env.make_state(np.array([9,9]))

    max_steps = 10
    goal_bias = 0.1

    rsg = RandomSampleGeneration(env)
    # rsg = RRT(env)
    path = rsg.search(start, target, max_steps=1000, goal_bias=0.1)
    rsg.draw_tree(plt.gca(), path=path, hold=True)
    
    smoothed_path = smooth_path(env, path)
    rsg.draw_tree(plt.gca(), path=smoothed_path, hold=True)

    # rrt.draw_voronoi_diagram()
