import numpy as np

from motion_planning.search import PRM
from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState, Path


class LazyPRM(PRM):
    def __init__(self, env: RobotSpace, num_samples: int =10, num_neighbors: int =10):
        super().__init__(env, num_samples=num_samples, num_neighbors=num_neighbors, validate_edges=False)
    
    def lazy_search(self, start: NumpyState, target: NumpyState, max_iter: int = 1000):
        i = 0
        
        while i < max_iter:

            potential_path = self.graph.dijkstra_search(start=start.value, end=target.value)
            potential_path_numpy = np.array(potential_path)

            start_edge_states = potential_path_numpy[:-1, :]
            end_edge_states = potential_path_numpy[1: , :]

            path_edge_validities = self.env.batch_is_valid_edge(start_states=start_edge_states, end_states=end_edge_states)
            invalid_edge_idxes = np.where(path_edge_validities == False)[0]

            if len(invalid_edge_idxes) == 0: 
                return potential_path

            for idx in invalid_edge_idxes:
                start_edge_state = start_edge_states[idx]
                end_edge_state = end_edge_states[idx]

                start_edge_idx = self.graph.vertex_to_idx[tuple(start_edge_state)]
                end_edge_idx = self.graph.vertex_to_idx[tuple(end_edge_state)]

                self.graph.edges[start_edge_idx].remove(end_edge_idx)
        
            i += 1
        
    def search(self, start, target):
        self.start = start 
        self.target = target

        # Attach Start and Target to the graph
        self.graph.add_vertex(start.value)
        self.graph.add_vertex(target.value)

        # Solve with A* or Dijkstra's Algorithm
        path = self.lazy_search(start=start, target=target)
        if path:
            path = Path([self.env.make_state(p) for p in path])

        # Return final Path
        return path

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from motion_planning.space import PolygonalRobot
    from motion_planning.obstacle_sets import RandomSamplePassage
    from motion_planning.utils import set_numpy_seed

    set_numpy_seed()

    env = PolygonalRobot()
    env.set_obstacles(RandomSamplePassage(num_walls=3, gap_width=2))

    prm = LazyPRM(env, num_samples=10000, num_neighbors=10)
    prm.create_graph()

    start, target = env.make_state(np.array([5.0, 5.0, 0.0])), env.make_state(np.array([35.0, 5.0, 0.0]))

    path = prm.search(start, target)

    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path, show_task=True)
    plt.show()