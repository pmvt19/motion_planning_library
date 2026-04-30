import numpy as np
import matplotlib.pyplot as plt

from motion_planning.space import ApproximationSpace
from motion_planning.space import PlanarMobileArm
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.utils import set_numpy_seed



if __name__ == '__main__':
    set_numpy_seed(0)

    env = PlanarMobileArm()
    env.set_obstacles(BiasedPassage())
    approx_env = ApproximationSpace(env, do_overapproximation=True)

    state = env.make_state(np.array([5.5, 5.0, np.pi/2, np.pi+(np.pi/4), np.pi+(np.pi/4)]))

    # Draw Base Env
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.savefig('saves/environments/circle_approx_base.png')

    # Draw Circle Approx Env
    plt.clf()
    approx_env.draw_environment(plt.gca())
    approx_env.draw_state(plt.gca(), state)
    plt.savefig('saves/environments/circle_approx.png')

    under_approx_env = ApproximationSpace(env, do_overapproximation=False)
    over_approx_env = ApproximationSpace(env, do_overapproximation=True)

    # TODO: Make these side-by-side plots?
    # Draw Circle Approx Env - Underapproximation
    plt.clf()
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state, color='black')
    under_approx_env.draw_state(plt.gca(), state)
    under_approx_env.draw_environment(plt.gca())
    plt.savefig('saves/environments/circle_approx_under.png')

    # Draw Circle Approx Env - Overapproximation
    plt.clf()
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state, color='black')
    over_approx_env.draw_state(plt.gca(), state)
    over_approx_env.draw_environment(plt.gca())
    plt.savefig('saves/environments/circle_approx_over.png')



    

