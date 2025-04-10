import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from shapely import Polygon, Point
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from space import PointRobot, PolygonalRobot, PlanarMobileArm
from matplotlib.collections import LineCollection
from state import NumpyState
from utils import smooth_path
from path import Path
from obstacle_sets import TestSet

class RRT():
    def __init__(self, env, delta=0.5):
        self.env = env
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
            new_node = self.env.shoot_ray(node, sampled_point, self.delta)
            # print(new_node.value)

        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node

        return new_node

    def draw_tree(self, ax, path:Path = None, show_task=True):
        self.env.draw_environment(ax)
        nodes = np.array([node.value for node in self.tree.keys()])
        # print(nodes)
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

    def step_search(self, goal_bias):
        exp_node, sampled_point = self.select_node(goal_bias=goal_bias)
        cur_node = self.expand_node(exp_node, sampled_point)
        return cur_node

    def search(self, start, target, max_steps=10000, goal_bias=0.1, animate_search_tree=False):
        self.init_search(start, target)

        cur_node = start
        num_steps = 0

        start_time = time.time()
        while (cur_node != target and num_steps < max_steps):
            print(f"Searching Step: {num_steps}", end='\r')
            cur_node = self.step_search(goal_bias=goal_bias)
            num_steps += 1
            if animate_search_tree:
                self.draw_tree(plt.gca())
                plt.pause(self.animation_delay)
                plt.clf()
        search_time = time.time() - start_time
        print(f"Search Time: {search_time}, Collision Checks: {self.env.num_collision_checks}")
        path = self.backtrack(end=target)
        
        return path

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
    

class BiDirectionalRRT():
    def __init__(self, env, delta=0.5, max_connection_distance=1):
        self.env = env
        self.delta = delta
        self.max_connection_distance = max_connection_distance

        self.animation_delay = 0.01

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
            return self.env.is_valid_edge(self.env.make_state(forward_tree_nodes[forward_tree_node_idx]), self.env.make_state(backward_tree_nodes[backward_tree_node_idx])), \
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

    def search(self, start, target, max_steps=10000, goal_bias=0.1, animate_search_tree=False):
        self.forward_rrt = RRT(env=self.env, delta=self.delta)
        self.backward_rrt = RRT(env=self.env, delta=self.delta)

        self.forward_rrt.init_search(start, target)
        self.backward_rrt.init_search(target, start)

        num_steps = 0
        is_connected = False
        start_time = time.time()

        while not is_connected and num_steps < max_steps:
            print(f'Searching Step: {num_steps}', end='\r')
            self.forward_rrt.step_search(goal_bias=goal_bias)
            self.backward_rrt.step_search(goal_bias=goal_bias)
            is_connected, connection = self.attempt_tree_connection(self.forward_rrt, self.backward_rrt)
            num_steps += 1

            if animate_search_tree:
                self.draw_tree(plt.gca())
                plt.pause(self.animation_delay)
                plt.clf()
        
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
    # seed = 22
    # seed = 18
    # seed = 81
    # seed = 5
    # seed = 25
    # seed = 6
    # seed = 66
    seed = 95
    print(f"Setting Seed: {seed}")
    
    np.random.seed(seed)
    start = (0, 0)
    target = (9, 9)
    
    # env = OpenSpace2d()
    # env = Environment2d()
    # env = RandomSamplePassage()
    # env = CarParkingEnv()

    env = PlanarMobileArm(num_links=3)
    env.set_obstacles(TestSet())
    # env = PointRobot()
    # env = PolygonalRobot()

    # env = PlanarMobileArm(num_links=4)
    start, target = env.sample_task()

    # start = env.make_state(np.array([0.0,0.0]))
    # start = env.make_state(np.array([-9.0,-9.0]))
    # target = env.make_state(np.array([9.0,9.0]))
    # target = env.make_state(np.array([15.0, 2.0]))
    # target = env.make_state(np.array([35.0, 2.0]))

    max_steps = 10
    goal_bias = 0.1

    # rrt = RRT(env)
    # rrt = RRTStar(env)
    rrt = BiDirectionalRRT(env)
    # path = rrt.search(start, target, max_steps=150, goal_bias=0.1, animate_search_tree=False)
    # path = rrt.search(start, target, max_steps=750, goal_bias=0.1, animate_search_tree=False)
    # path = rrt.search(start, target, max_steps=1000, goal_bias=0.1, animate_search_tree=False)
    path = rrt.search(start, target, max_steps=6000, goal_bias=0.1, animate_search_tree=False)
    # path = rrt.search(start, target, max_steps=3000, do_rewire=True, goal_bias=0.4, animate_search_tree=False)
    # path = rrt.search(start, target, max_steps=4000, goal_bias=0.1, animate_search_tree=False)
    # path = rrt.search(start, target, max_steps=4000, goal_bias=0.1, animate_search_tree=False)
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()

    env.animate_path(path, frame_delay=0.1)
    
    # smoothed_path = smooth_path(env, path)
    # rrt.draw_tree(plt.gca(), path=smoothed_path)
    # plt.show()

    # env.animate_path(smoothed_path, frame_delay=0.1)



    # from kinematic_path_smoothing import smooth_path_trajectory_optimization
    # smoothed_path = smooth_path_trajectory_optimization(env, path)

    # rrt.draw_voronoi_diagram()
