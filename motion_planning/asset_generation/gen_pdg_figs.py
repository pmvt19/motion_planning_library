import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from motion_planning.database import Database
from motion_planning.search import PDG
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.space import ApproximationSpace

def draw_bpe3_env(save_fig=False):
    np.random.seed(0)
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))

    env.draw_environment(plt.gca())
    
    if save_fig:
        plt.savefig("saves/pdg/environment.png")
    else:
        plt.show()

def draw_path_database(path='saves/database_bpe3_large.pickle', save_fig=False):
    np.random.seed(0)
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))


    db = pickle.load(open(path, 'rb'))

    for path in db:
        numpy_path = np.array([p.value for p in path.path])
        plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o')
    env.draw_environment(plt.gca())
    
    if save_fig:
        plt.savefig("saves/pdg/database.png")
    else:
        plt.show()

def draw_relevant_paths(path="saves/database_bpe3_large.pickle", save_fig=False):
    np.random.seed(0)
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))

    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = PDG(env, path)
    target = env.make_state(np.array([15.0, 5.0]))
    pdg.compute_retained_paths(target)

    validated_paths = pdg.validated_paths

    for path in validated_paths:
        numpy_path = np.array([p.value for p in path.path])
        plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o')
    env.space.draw_environment(plt.gca())
    plt.scatter(target.value[0], target.value[1], s=100, color='red', zorder=2)
    
    if save_fig:
        plt.savefig("saves/pdg/relevant_paths.png")
    else:
        plt.show()

def draw_search_tree_steps(path="saves/database_bpe3_large.pickle", save_fig=False):
    np.random.seed(0)
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))

    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = PDG(env, path)

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([15.0, 5.0]))

    pdg.compute_retained_paths(target)

    pdg.init_search(start, target)

    path = None
    for i in range(500):
        plt.cla()
        pdg.step_search(i)
        pdg.draw_tree(plt.gca())

        if target in pdg.tree:
            path = pdg.backtrack(target)
            break
        
        if save_fig:
            plt.savefig(f"saves/pdg/step_{i}.png")
        else:
            plt.pause(0.1)
        
    
    if save_fig:
        plt.savefig(f"saves/pdg/final.png")
    else:
        plt.show()
    

    plt.cla()
    pdg.draw_tree(plt.gca(), path=path)
    if save_fig:
        plt.savefig(f"saves/pdg/final_path.png")
    else:
        plt.show()
    
draw_bpe3_env(save_fig=True)
# draw_path_database(save_fig=True)
# draw_relevant_paths(save_fig=True)
# draw_search_tree_steps(save_fig=True)