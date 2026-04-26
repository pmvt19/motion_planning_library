import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
from matplotlib.collections import LineCollection

from motion_planning.state import NumpyState
from motion_planning.utils import smooth_path
from motion_planning.rrt import RRT 
from motion_planning.space import PointRobot, PolygonalRobot, PlanarMobileArm
from motion_planning.obstacle_sets import BiasedPassage

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

            dists = [self.env.dist(self.env.make_state(new_node_candidates[i]), sampled_point) if (self.env.is_valid(new_node_candidates[i]) and self.env.is_valid_edge(node, self.env.make_state(new_node_candidates[i]))) else float('inf') for i in range(self.num_neighbors)]

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
    seed = np.random.randint(0, 10000)
    seed = 54
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)

    env = PointRobot()
    env.set_obstacles(BiasedPassage())
    start, target = env.sample_task()

    rsg = RandomSampleGeneration(env)

    path = rsg.search(start, target, max_steps=1000, goal_bias=0.1)
    rsg.draw_tree(plt.gca(), path=path)
    plt.show()

