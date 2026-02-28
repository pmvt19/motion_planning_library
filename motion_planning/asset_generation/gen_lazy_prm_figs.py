import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from heapq import heappush, heappop

from motion_planning.prm import LazyPRM
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import RandomSamplePassage, BiasedPassage
from motion_planning.utils import set_numpy_seed
from motion_planning.state import NumpyState
from motion_planning.path import Path

def draw_invalid_edges(ax, edges):
    # ax.add_collection(LineCollection(edges, color='#a61107'))
    ax.add_collection(LineCollection(edges, color="#c00789"))

def lazy_search(prm, start: NumpyState, target: NumpyState, max_iter: int = 1000, save_figs: bool = False):
        i = 0
        
        while i < max_iter:

            potential_path = prm.graph.dijkstra_search(start=start.value, end=target.value)
            potential_path_numpy = np.array(potential_path)

            start_edge_states = potential_path_numpy[:-1, :]
            end_edge_states = potential_path_numpy[1: , :]

            path_edge_validities = prm.env.batch_is_valid_edge(start_states=start_edge_states, end_states=end_edge_states)
            invalid_edge_idxes = np.where(path_edge_validities == False)[0]

            potential_path_wrapped = Path([prm.env.make_state(p) for p in potential_path])
            prm.env.draw_environment(plt.gca())
            invalid_edges = [(start_edge_states[idx], end_edge_states[idx]) for idx in invalid_edge_idxes]
            prm.draw(plt.gca(), path=potential_path_wrapped, show_task=True)
            draw_invalid_edges(plt.gca(), invalid_edges)

            if save_figs:
                plt.savefig(f'saves/lazy_prm/step_{i}.png')
                plt.clf()
            else:
                plt.show()

            if len(invalid_edge_idxes) == 0: 
                return potential_path

            for idx in invalid_edge_idxes:
                start_edge_state = start_edge_states[idx]
                end_edge_state = end_edge_states[idx]

                start_edge_idx = prm.graph.vertex_to_idx[tuple(start_edge_state)]
                end_edge_idx = prm.graph.vertex_to_idx[tuple(end_edge_state)]

                prm.graph.edges[start_edge_idx].remove(end_edge_idx)
        
            i += 1

def search(prm, start, target, save_figs=False):
    prm.start = start 
    prm.target = target

    # Attach Start and Target to the graph
    prm.graph.add_vertex(start.value)
    prm.graph.add_vertex(target.value)

    # Solve with A* or Dijkstra's Algorithm
    path = lazy_search(prm, start=start, target=target, save_figs=save_figs)
    if path:
        path = Path([prm.env.make_state(p) for p in path])
    # Return final Path
    return path

if __name__ == '__main__':
    seed = 2433 
    set_numpy_seed(seed)

    # Create Robot Environment with Random Sample Passage Obstacles
    env = PointRobot()
    os = BiasedPassage()
    env.set_obstacles(os)

    prm = LazyPRM(env, num_samples=100, num_neighbors=20)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([2.5, 5.0])), env.make_state(np.array([17.5, 5.0]))

    prm.create_graph()
    path = search(prm, start, target, save_figs=True)

    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path)
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/lazy_prm/lazy_prm.png')
    # plt.show()

