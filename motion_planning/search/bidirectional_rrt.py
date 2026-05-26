import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

from motion_planning.search import RRT
from motion_planning.tools import Path


class BiDirectionalRRT:
    def __init__(self, env, delta=0.5, max_connection_distance=1):
        self.env = env
        self.delta = delta
        self.max_connection_distance = max_connection_distance

        self.animation_delay = 0.01

    def attempt_tree_connection(self, forward_rrt, backward_rrt):
        forward_tree_nodes = np.array([node.value for node in forward_rrt.tree.keys()])
        backward_tree_nodes = np.array(
            [node.value for node in backward_rrt.tree.keys()]
        )

        kdt = KDTree(forward_tree_nodes)
        dist, ind = kdt.query(backward_tree_nodes, k=1)
        dist = dist.flatten()
        ind = ind.flatten()

        if np.min(dist) < self.max_connection_distance:
            backward_tree_node_idx = np.argmin(dist)
            forward_tree_node_idx = ind[backward_tree_node_idx]
            return self.env.is_valid_edge(
                self.env.make_state(forward_tree_nodes[forward_tree_node_idx]),
                self.env.make_state(backward_tree_nodes[backward_tree_node_idx]),
            ), (
                self.env.make_state(forward_tree_nodes[forward_tree_node_idx]),
                self.env.make_state(backward_tree_nodes[backward_tree_node_idx]),
            )
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

    def search(
        self, start, target, max_steps=10000, goal_bias=0.1, animate_search_tree=False
    ):
        self.forward_rrt = RRT(env=self.env, delta=self.delta)
        self.backward_rrt = RRT(env=self.env, delta=self.delta)

        self.forward_rrt.init_search(start, target)
        self.backward_rrt.init_search(target, start)

        num_steps = 0
        is_connected = False
        start_time = time.time()

        while not is_connected and num_steps < max_steps:
            print(f"Searching Step: {num_steps}", end="\r")
            self.forward_rrt.step_search(goal_bias=goal_bias)
            self.backward_rrt.step_search(goal_bias=goal_bias)
            is_connected, connection = self.attempt_tree_connection(
                self.forward_rrt, self.backward_rrt
            )
            num_steps += 1

            if animate_search_tree:
                self.draw_tree(plt.gca())
                plt.pause(self.animation_delay)
                plt.clf()

        search_time = time.time() - start_time
        print(
            f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}"
        )

        path = self.backtrack(
            forward_rrt=self.forward_rrt,
            backward_rrt=self.backward_rrt,
            connection=connection,
        )
        return path

    def draw_tree(self, ax, path: Path = None, hold=False, show_task=True):
        self.env.draw_environment(ax)
        self.backward_rrt.draw_tree(ax, show_task=False)
        self.forward_rrt.draw_tree(ax, path=path, show_task=show_task)


if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))
    rrt = BiDirectionalRRT(env)

    start, target = (
        env.make_state(np.array([5.0, 5.0])),
        env.make_state(np.array([15.0, 5.0])),
    )
    path = rrt.search(start, target, max_steps=7000)

    rrt.draw_tree(plt.gca(), path=path)
    plt.show()
