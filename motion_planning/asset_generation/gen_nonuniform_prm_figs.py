import numpy as np
import matplotlib.pyplot as plt

from motion_planning.prm import NonUniformPRM
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import RandomSamplePassage
from motion_planning.utils import set_numpy_seed

if __name__ == '__main__':
    seed = 2433 
    set_numpy_seed(seed)

    # Create Robot Environment with Random Sample Passage Obstacles
    env = PointRobot()
    os = RandomSamplePassage(num_walls=2)
    env.set_obstacles(os)

    prm = NonUniformPRM(env, num_samples=1000, num_neighbors=10, validate_edges=True)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([6.5, 5.0])), env.make_state(np.array([26.5, 5.0]))

    prm.create_graph()
    path = prm.search(start, target)

    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path)
    plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
    plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
    plt.savefig('saves/nonuniform_prm/nonuniform_prm.png')

