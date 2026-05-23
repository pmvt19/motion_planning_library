import matplotlib.pyplot as plt
import numpy as np

from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.search import RandomSampleGeneration
from motion_planning.space import PointRobot
from motion_planning.utils import set_numpy_seed

if __name__ == "__main__":
    set_numpy_seed(1682)

    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    env.set_obstacles(os)

    rsg = RandomSampleGeneration(env)

    # Assign Fixed Start and Target Positions
    start, target = (
        env.make_state(np.array([2.5, 5.0])),
        env.make_state(np.array([17.5, 5.0])),
    )

    path = rsg.search(start, target, max_steps=4000, goal_bias=0.08)

    # Draw Final Tree with Path
    rsg.draw_tree(plt.gca(), path=path)
    plt.savefig("saves/rsg/rsg_tree.png")
