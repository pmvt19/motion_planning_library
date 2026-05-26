import matplotlib.pyplot as plt
import numpy as np

from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.search import RRT
from motion_planning.space import PointRobot
from motion_planning.utils import set_numpy_seed


# Custom Search Function to Save Each Step During Search
def search_and_save(
    rrt: RRT, start, target, max_steps=10000, goal_bias=0.1, starting_tree_info=None
):
    rrt.init_search(start, target, starting_tree_info)

    cur_node = start
    num_steps = 0

    while cur_node != target and num_steps < max_steps:
        print(f"Searching Step: {num_steps}", end="\r")
        cur_node = rrt.step_search(goal_bias=goal_bias)

        rrt.draw_tree(plt.gca())
        plt.savefig(f"saves/rrt/rrt_steps/step_{num_steps}.png")
        plt.clf()
        num_steps += 1
    print(f"Found in {num_steps}")
    path = rrt.backtrack(end=target)

    return path


if __name__ == "__main__":
    set_numpy_seed(0)

    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    env.set_obstacles(os)

    rrt = RRT(env)

    # Assign Fixed Start and Target Positions
    start, target = (
        env.make_state(np.array([2.5, 5.0])),
        env.make_state(np.array([17.5, 5.0])),
    )

    # Use Custom Search Function To Search and Save Steps
    # Caution: Can Generate Large Files
    path = search_and_save(rrt, start, target, max_steps=4000, goal_bias=0.08)

    # Draw Final Tree with Path
    plt.clf()
    rrt.draw_tree(plt.gca(), path=path)
    plt.savefig("saves/rrt/rrt_steps/final.png")

    rrt = RRT(env, delta=1.0)
    rrt.search(start, target, max_steps=25, goal_bias=0.0)
    rrt.draw_voronoi_diagram()
    plt.savefig("saves/rrt/voronoi_diagram.png")
