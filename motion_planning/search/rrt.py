import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KDTree

from motion_planning.space import RobotSpace
from motion_planning.tools import Path


class RRT:
    def __init__(self, env, delta=0.5):
        self.env: RobotSpace = env
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        self.delta = delta

        self.animation_delay = 0.01

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
            # print(np.linalg.norm(sampled_point.value - node.value), 'dist')
            # new_node = self.env.shoot_ray(node, sampled_point, self.delta)
            new_node = self.env.shoot_ray(
                node,
                sampled_point,
                min(self.delta, np.linalg.norm(sampled_point.value - node.value)),
            )
            # assert(new_node.value[1] < 10), "ERROR EXPAND NODE"
        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node

        return new_node

    def draw_tree(self, ax, path: Path = None, show_task=True):
        self.env.draw_environment(ax)
        nodes = np.array([node.value for node in self.tree.keys()])

        try:
            ax.scatter(nodes[:, 0], nodes[:, 1])
            if show_task:
                ax.scatter(self.start.value[0], self.start.value[1], s=100, c="green")
                ax.scatter(self.target.value[0], self.target.value[1], s=100, c="red")

            edges = [
                [(p.value[0], p.value[1]), (c.value[0], c.value[1])]
                for p in self.tree
                for c in self.tree[p]
            ]
            edges = LineCollection(edges, color="blue")
            ax.add_collection(edges)

        except Exception:
            FAIL = "\033[91m"
            ENDC = "\033[0m"
            print(f"{FAIL}Cannot Draw Tree Without Running Search{ENDC}")
            raise Exception

        if path:
            path = [
                (path[i].value[:2], path[i + 1].value[:2]) for i in range(len(path) - 1)
            ]
            ax.add_collection(LineCollection(path, color="red"))

    def draw_voronoi_diagram(self):
        nodes = np.array([node.value for node in self.tree.keys()])
        vor = Voronoi(nodes)
        fig = voronoi_plot_2d(
            vor,
            show_vertices=False,
            line_colors="orange",
            line_width=2,
            line_alpha=0.6,
            point_size=2,
        )
        self.draw_tree(fig.gca())

    def backtrack(self, end=None):
        if end is None or end not in self.child_to_parent:
            return Path()

        path = []
        node = end
        while node:
            path.append(node)
            node = self.child_to_parent[node]
        return Path(path=path[::-1])

    def init_search(self, start, target, starting_tree_info=None):
        self.start = start
        self.target = target

        # TODO: Clean This up with constructor (Semi-HACK)
        self.tree = defaultdict(list)
        self.child_to_parent = {}
        # TODO: Clean This up with constructor (Semi-HACK)

        self.tree[self.start] = []
        self.child_to_parent[self.start] = None

        if starting_tree_info:
            self.tree = starting_tree_info[0]
            self.child_to_parent = starting_tree_info[1]

    def step_search(self, goal_bias):
        exp_node, sampled_point = self.select_node(goal_bias=goal_bias)
        cur_node = self.expand_node(exp_node, sampled_point)
        return cur_node

    def search(
        self,
        start,
        target,
        max_steps=10000,
        goal_bias=0.1,
        animate_search_tree=False,
        starting_tree_info=None,
    ):
        self.init_search(start, target, starting_tree_info)

        cur_node = start
        num_steps = 0

        start_time = time.time()
        while cur_node != target and num_steps < max_steps:
            print(f"Searching Step: {num_steps}", end="\r")
            cur_node = self.step_search(goal_bias=goal_bias)
            num_steps += 1
            if animate_search_tree:
                self.draw_tree(plt.gca())
                plt.pause(self.animation_delay)
                plt.clf()
        search_time = time.time() - start_time
        print(
            f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}"
        )
        path = self.backtrack(end=target)

        return path


if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import DiscRobot

    seed = np.random.randint(0, 100)
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)

    env = DiscRobot(disc_radius=0.3)
    env.set_obstacles(BiasedPassage(num_walls=1))
    start, target = (
        env.make_state(np.array([2.5, 5.0])),
        env.make_state(np.array([17.5, 5.0])),
    )

    rrt = RRT(env)
    path = rrt.search(
        start, target, max_steps=4000, goal_bias=0.0, animate_search_tree=False
    )

    rrt.draw_tree(plt.gca(), path, show_task=True)
    plt.show()

    env.animate_path(path)
