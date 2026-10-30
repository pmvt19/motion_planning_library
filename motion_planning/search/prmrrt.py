import numpy as np

from matplotlib.collections import LineCollection
from sklearn.neighbors import KDTree

from motion_planning.search import PRM
from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState, Tree, Path


class PRMRRT(PRM):
    def __init__(
        self,
        env: RobotSpace,
        num_samples=10,
        num_neighbors=None,
        edge_dist_radius=None,
        validate_edges=False,
    ):
        super().__init__(
            env=env, 
            num_samples=num_samples,
            num_neighbors=num_neighbors,
            edge_dist_radius=edge_dist_radius,
            validate_edges=validate_edges
        )

        self.tree = Tree()

    def select_node(self, goal_bias=0.1):
        if np.random.random() < goal_bias:
            sampled_point = self.target
        else:
            sampled_point = self.env.sample_valid_point()

        # print([node.value for node in self.tree.tree.keys()])
        nodes = np.array([node.value for node in self.tree.tree.keys()])
        # print("Node: ", nodes.shape)
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point

    def step_search(self, iteration: int):
        # Sample Point
        node, sampled_point = self.select_node()

        # Get Neighbors of Chosen Node via Graph functions
        neighbors_tuple = self.graph.get_neighbors(tuple(node.value))
        # print(neighbors_tuple)

        # Convert Graph Neighbors to type: NumpyState
        neighbors_state = [self.env.make_state(state) for state in neighbors_tuple]

        # Extend to the closest neighbor to the sampled point
        ordered_dists = [self.env.dist(sampled_point, neighbor) for neighbor in neighbors_state]
        min_idx = np.argmin(ordered_dists)
        potential_new_child = neighbors_state[min_idx]
        if potential_new_child in self.tree.tree:
            return None
        else:
            self.tree.add_node(parent_node=node, child_node=potential_new_child)
            return potential_new_child

    def init_search(self, start: NumpyState, target: NumpyState):
        self.start = start
        self.target = target

        self.tree.init_tree(self.start)
        
        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

    def search(self, start: NumpyState, target: NumpyState, max_steps: int = 1000):
        self.init_search(start, target)

        step_num = 0
        cur_node = start
        while cur_node != target and step_num < max_steps:
            potential_cur_node = self.step_search(step_num)
            if potential_cur_node:
                cur_node = potential_cur_node
            step_num += 1

        reversed_path = self.tree.backtrack(target)
        return Path(reversed_path[::-1])

    def draw(self, ax, path: Path | None, show_task: bool = True):
        self.env.draw_environment(ax)

        self.graph.draw(ax)
        self.tree.draw_tree(ax)

        if show_task:
            ax.scatter(self.start.value[0], self.start.value[1], s=100, color="green")
            ax.scatter(self.target.value[0], self.target.value[1], s=100, color="red")

        if path:
            path = [
                (path[i].value[:2], path[i + 1].value[:2]) for i in range(len(path) - 1)
            ]
            ax.add_collection(LineCollection(path, color="red"))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    env = PointRobot()
    env.set_obstacles(BiasedPassage())

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([15.0, 5.0]))

    prmrrt = PRMRRT(env=env, num_samples=500, num_neighbors=10, validate_edges=True)
    prmrrt.create_graph()
    path = prmrrt.search(start, target)

    prmrrt.draw(plt.gca(), path)
    plt.show()
            

        