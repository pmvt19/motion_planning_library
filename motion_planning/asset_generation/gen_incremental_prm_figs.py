import numpy as np
import matplotlib.pyplot as plt

from motion_planning.search import PRM, IncrementalPRM
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.utils import set_numpy_seed
from motion_planning.tools import Graph, Path

def search_and_save(prm, env, start, target, save_figs=False):
    prm.start = start
    prm.target = target

    prm.create_graph()

    # Attach Start and Target to the graph
    prm.graph.add_vertex(start.value)
    prm.graph.add_vertex(target.value)

    prm.batch_validate_graph_edges()
    is_connected = prm.is_nodes_connected(start, target)
    num_times_extended = 0
    while not is_connected:
        print(f"Adding Nodes, Current Size: {len(prm.graph.vertices)}")
        prm.extend_graph()
        prm.batch_validate_graph_edges()
        # plt.clf()
        env.draw_environment(plt.gca())
        prm.draw(plt.gca())
        plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
        plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
        if save_figs:
            plt.savefig(f"saves/incremental_prm/step_{num_times_extended}.png")
        else:
            plt.show()
        is_connected = prm.is_nodes_connected(start, target)
        num_times_extended += 1

    print(f"Num Times Extended: {num_times_extended}")

    # Solve with A* or Dijkstra's Algorithm
    path = prm.graph.dijkstra_search(start=start.value, end=target.value)

    if path:
        path = Path([env.make_state(p) for p in path])

    return path

if __name__ == '__main__':
    seed = 9405 
    set_numpy_seed(seed=seed)

    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    env.set_obstacles(os)

    prm = IncrementalPRM(env, num_samples=10, num_neighbors=5)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([2.5, 5.0])), env.make_state(np.array([17.5, 5.0]))

    # Visualize Empty Environment with Task
    env.draw_environment(plt.gca())
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/incremental_prm/initial_env.png')

    # Search via Incremental PRM and Save Steps
    path = search_and_save(prm, env, start, target, save_figs=True)

    # Visualize PRM Graph with Path:
    prm.draw(plt.gca(), path=path, show_task=True)
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/incremental_prm/graph_with_path.png')
