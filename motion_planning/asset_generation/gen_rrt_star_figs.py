import numpy as np
import matplotlib.pyplot as plt

from motion_planning.rrt import RRTStar, RRT
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.utils import set_numpy_seed

# IDEA:
# Make two side by sid graphs, one with RRT and the other with RRT*
# RRT will stop at the first path found while RRT* will continue searching beyond the initial path to improve the path

# IDEA ATTEMPT
# UNCLEAR WHICH METHOD TO USE TO CREATE THE ASSET
def rrt_and_rrt_star_compare(save_figs=False):
    set_numpy_seed(5025)

    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    # env.set_obstacles(os)

    
    rrt = RRT(env)
    rrt_star = RRTStar(env)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([0.0, 0.0])), env.make_state(np.array([5.0, 5.0]))

    # path = rrt.search(start, target, max_steps=1000, goal_bias=0.0, animate_search_tree=True)
    rrt.init_search(start, target, None)
    rrt_star.init_search(start, target)
    rrt_node = start

    rrt_path = None
    rrt_star_path = None

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(350):
        if rrt_node != target:
            rrt_node = rrt.step_search(goal_bias=0.0)
            if rrt_node == target:
                rrt_path = rrt.backtrack(end=target)
        
        rrt_star.step_search(rewire=True, goal_bias=0.0)

        rrt_star.draw_tree(ax[0])
        rrt.draw_tree(ax[1], path=rrt_path)
        
        ax[0].set_title("RRT*")
        ax[1].set_title("RRT")

        if save_figs:
            plt.savefig(f'saves/search/rrt_star/steps/step_{i}.png')
        else:
            if i == 0:
                plt.pause(5.0)
            else:
                plt.pause(0.1)
        ax[0].cla()
        ax[1].cla()

    rrt_star_path = rrt_star.backtrack(end=target)
    
    rrt_star.draw_tree(ax[0], path=rrt_star_path)
    rrt.draw_tree(ax[1], path=rrt_path)

    if save_figs:
        plt.savefig(f'saves/search/rrt_star/steps/final.png')
    else:
        plt.show()

def rrt_star_animation():
    # Create Robot Environment with Biased Passage Obstacles
    env = PointRobot()
    os = BiasedPassage(num_walls=1)
    # env.set_obstacles(os)

    rrt = RRTStar(env)

    # Assign Fixed Start and Target Positions
    start, target = env.make_state(np.array([0.0, 0.0])), env.make_state(np.array([5.0, 5.0]))

    path = rrt.search(start, target, max_steps=1000, goal_bias=0.0, animate_search_tree=True)

    # Draw Final Tree with Path
    plt.clf()
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()

# To Record a Video and Convert to GIF Online
if __name__ == '__main__':
    set_numpy_seed(5025)

    rrt_and_rrt_star_compare()
    # rrt_star_animation()

    


    



