import numpy as np
import matplotlib.pyplot as plt

from motion_planning.rrt import BiDirectionalRRT
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.utils import set_numpy_seed

if __name__ == '__main__':
    set_numpy_seed(8381)

    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    env.set_obstacles(os)

    rrt = BiDirectionalRRT(env)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([2.5, 5.0])), env.make_state(np.array([17.5, 5.0]))

    # Use Custom Search Function To Search and Save Steps
    # Caution: Can Generate Large Files
    path = rrt.search(start, target, max_steps=4000, goal_bias=0.01, animate_search_tree=True)

    # Draw Final Tree with Path
    plt.clf()
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()



