import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.neighbors import KDTree

from motion_planning.search import RRT
from motion_planning.tools import Path

class RRTStar(RRT):
    def __init__(self, env, delta=0.5, rewire_radius=1, max_rewire_neighbors=20):
        super().__init__(env=env, delta=delta)
        self.rewire_radius = rewire_radius
        self.max_rewire_neighbors = max_rewire_neighbors
    
    def init_search(self, start, target):
        self.start = start
        self.target = target
        self.tree[self.start] = []
        self.child_to_parent[self.start] = (None, 0)
    
    def expand_node(self, node, sampled_point):
        if self.target not in self.tree and self.env.dist(self.target.value, node.value) < self.delta:
            new_node = self.target
        else:
            new_node = self.env.shoot_ray(node, sampled_point, self.delta)
        
        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = (node, self.child_to_parent[node][1] + self.env.dist(node.value, new_node.value))
        return new_node
    
    def dfs_update(self, parent, cost):
        for child in self.tree[parent]:
            child_cost = cost + self.env.dist(parent.value, child.value)
            self.child_to_parent[child] = (parent, child_cost)
            self.dfs_update(child, child_cost)

    def update_child_nodes(self, parent, child):
        # Delete child from its current parent
        old_parent, _ = self.child_to_parent[child]
        self.tree[old_parent].remove(child)

        # Update Parent of Child
        self.tree[parent].append(child)
        child_cost = self.child_to_parent[parent][1] + self.env.dist(parent.value, child.value)
        self.child_to_parent[child] = (parent, child_cost)

        # Update Descendant Cost
        self.dfs_update(child, child_cost)


    def rewire(self, q_new):
        k = min(int(len(self.tree) * 1), self.max_rewire_neighbors)

        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        dists, ind = kdt.query(np.array([q_new.value]), k=k)
        # print(dists, ind)
        dists = dists[0][1:]
        ind = ind[0][1:]
        
        for i, idx in enumerate(ind):
            if dists[i] < self.rewire_radius:
                q = self.env.make_state(nodes[idx])
                cost_q = self.child_to_parent[q][1]
                cost_qnew = self.child_to_parent[q_new][1]

                distance = self.env.dist(q.value, q_new.value)
                qq_new_edge_validity = self.env.is_valid_edge(q, q_new)

                if (cost_qnew + distance < cost_q) and qq_new_edge_validity:
                    # optimal path to q passes through q_new
                    self.update_child_nodes(parent=q_new, child=q)

                if (cost_q + distance < cost_qnew) and qq_new_edge_validity:
                    # optimal path to q_new passes through q
                    self.update_child_nodes(parent=q, child=q_new)

    def step_search(self, rewire, goal_bias):
        exp_node, sampled_point = self.select_node(goal_bias=goal_bias)
        cur_node = self.expand_node(exp_node, sampled_point)
        if rewire:
            self.rewire(cur_node)
        return cur_node

    def backtrack(self, end=None):
        if end is None or end not in self.child_to_parent:
            return Path()

        path = []
        node = end
        while node:
            path.append(node)
            node = self.child_to_parent[node][0]
        return Path(path=path[::-1])
    

    def search(self, start, target, max_steps=10000, goal_bias=0.1, do_rewire=True, animate_search_tree=False):
        self.init_search(start, target)

        num_steps = 0
        start_time = time.time()
        while (num_steps < max_steps):
            print(f"Searching Step: {num_steps}", end='\r')
            cur_node = self.step_search(rewire=do_rewire, goal_bias=goal_bias)
            num_steps += 1
            if animate_search_tree:
                self.draw_tree(plt.gca())
                plt.pause(self.animation_delay)
                plt.clf()

        search_time = time.time() - start_time
        print(f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}")
        path = self.backtrack(end=target)
        return path

if __name__ == '__main__':
    from motion_planning.space import PointRobot
    from motion_planning.obstacle_sets import BiasedPassage

    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))
    rrt = RRTStar(env)

    start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))
    path = rrt.search(start, target)

    rrt.draw_tree(plt.gca(), path=path)
    plt.show()
    