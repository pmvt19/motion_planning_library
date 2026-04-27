import numpy as np
import matplotlib.pyplot as plt

from motion_planning.search import PRM
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.utils import set_numpy_seed
from motion_planning.tools import Graph

    
def visualize_prm_vertices():
    pass

def visualize_prm_graph_with_invalid_edges():
    pass

def visualize_prm_graph_with_invalid_edges():
    pass


if __name__ == '__main__':
    seed = 9405 
    set_numpy_seed(seed=seed)

    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    env.set_obstacles(os)

    prm = PRM(env, num_samples=50, num_neighbors=10, validate_edges=False)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([2.5, 5.0])), env.make_state(np.array([17.5, 5.0]))

    prm.create_graph(starting_samples=[start.value, target.value])

    # Visualize Empty Environment with Task
    env.draw_environment(plt.gca())
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/prm/initial_env.png')

    # Visualize All Vertices
    env.draw_environment(plt.gca())
    plt.scatter(prm.graph.vertices[:, 0], prm.graph.vertices[:, 1], color='#FFA500')
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/prm/vertices_only.png')

    # Visualize All Vertices and Potential Edges
    env.draw_environment(plt.gca())
    prm.graph.draw(plt.gca())
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/prm/graph_with_invalid_edges.png')

    # Validate Edges and Create Path:
    prm.batch_validate_graph_edges()
    path = prm.search(start, target)
    
    # Visualize All Vertices and Edges and Mark Invalid Edges
    prm.draw(plt.gca(), plot_invalid_edges=True, show_task=True)
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/prm/graph_with_marked_invalid_edges.png')

    # Visualize ALl Vertices and Valid Edges
    plt.clf()
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), plot_invalid_edges=False, show_task=True)
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/prm/graph_with_marked_valid_edges.png')

    # Visualize PRM Graph with Path:
    prm.draw(plt.gca(), path=path, show_task=True)
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/prm/graph_with_path.png')
