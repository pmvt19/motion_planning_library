import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

from motion_planning.search import PDG
from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState, Path


class BiDirectionalPDG:
    def __init__(self, env: RobotSpace, db_path: str):
        self.env: RobotSpace = env
        self.db_path: str = db_path
        self.max_connection_distance: float = 0.5

        start_time = time.time()
        self.forward_pdg: PDG = PDG(self.env, self.db_path)
        self.backward_pdg: PDG = PDG(self.env, self.db_path)
        end_time = time.time()
        print(f"Time to Instantiate PDGs: {end_time - start_time}")

    def attempt_tree_connection(
        self, forward_pdg: PDG, backward_pdg: PDG
    ) -> tuple[bool, tuple[NumpyState, NumpyState]]:
        forward_tree_nodes = np.array([node.value for node in forward_pdg.tree.keys()])
        backward_tree_nodes = np.array(
            [node.value for node in backward_pdg.tree.keys()]
        )

        kdt = KDTree(forward_tree_nodes)
        dist, ind = kdt.query(backward_tree_nodes, k=1)
        dist = dist.flatten()
        ind = ind.flatten()

        if np.min(dist) < self.max_connection_distance:
            backward_tree_node_idx = np.argmin(dist)
            forward_tree_node_idx = ind[backward_tree_node_idx]

            forward_tree_node_np = forward_tree_nodes[forward_tree_node_idx]
            backward_tree_node_np = backward_tree_nodes[backward_tree_node_idx]

            forward_tree_node = self.env.make_state(forward_tree_node_np)
            backward_tree_node = self.env.make_state(backward_tree_node_np)

            connection_edge_validity = self.env.is_valid_edge(
                forward_tree_node, backward_tree_node
            )

            return connection_edge_validity, (forward_tree_node, backward_tree_node)
        return False, None

    # TODO: Can use the util function
    def delete_duplicate_states_in_path(
        self, path: list[NumpyState]
    ) -> list[NumpyState]:
        i = len(path) - 1
        while i > 0:
            if np.all(path[i - 1] == path[i]):
                path.pop(i)
            i -= 1
        return path

    def backtrack(
        self,
        forward_pdg: PDG,
        backward_pdg: PDG,
        connection: tuple[NumpyState, NumpyState],
    ) -> Path:
        start_time = time.time()
        if connection is None:
            return Path()
        forward_end_state, backward_end_state = connection
        forward_path = forward_pdg.backtrack(end=forward_end_state)
        backward_path = backward_pdg.backtrack(end=backward_end_state)

        # Join Both Paths and Reverse the backward PDG Tree path
        joined_path = forward_path.path + backward_path.path[::-1]
        joined_path = self.delete_duplicate_states_in_path(joined_path)
        end_time = time.time()
        print(f"Time to backtrack path: {end_time - start_time}")
        return Path(path=joined_path)

    def search(
        self, start: NumpyState, target: NumpyState, max_steps: int = 500
    ) -> Path:
        self.start: NumpyState = start
        self.target: NumpyState = target

        start_time = time.time()
        self.forward_pdg.compute_retained_paths(self.target)
        self.backward_pdg.compute_retained_paths(self.start)
        end_time = time.time()
        print(f"Time to Compute Retained Paths: {end_time - start_time}")

        start_time = time.time()
        self.forward_pdg.init_search(start, target)
        self.backward_pdg.init_search(target, start)
        end_time = time.time()
        print(f"Time to Initialize Search: {end_time - start_time}")

        st = time.time()
        for i in range(max_steps):
            self.forward_pdg.step_search(i)
            self.backward_pdg.step_search(i)
            is_connected, connection = self.attempt_tree_connection(
                self.forward_pdg, self.backward_pdg
            )

            if is_connected:
                break
        et = time.time()
        print(f"Time to search (without computing retained paths): {et - st}")

        if is_connected:
            return self.backtrack(self.forward_pdg, self.backward_pdg, connection)
        else:
            return Path([])

    def draw_tree(self, ax: plt.Axes, path: Path | None = None, show_task: bool = True):
        self.env.draw_environment(ax)
        self.backward_pdg.draw_tree(ax=ax, path=None, show_task=False)
        self.forward_pdg.draw_tree(ax=ax, path=path, show_task=show_task)


if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import ApproximationSpace, PointRobot

    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    db_save_path = "saves/database_pr_bp3_large.pickle"

    bidirectional_pdg = BiDirectionalPDG(env, db_save_path)

    start, target = env.space.sample_task()

    start_time = time.time()
    path = bidirectional_pdg.search(start, target)
    end_time = time.time()
    time_to_search = end_time - start_time
    print(f"Time to search (BiDir): {time_to_search}")

    plt.cla()
    bidirectional_pdg.draw_tree(plt.gca(), path)
    plt.show()
