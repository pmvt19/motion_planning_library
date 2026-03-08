import pickle

import numpy as np
import matplotlib.pyplot as plt

from motion_planning.pdg import PDG
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.circle_approximation import ApproximationSpace

def draw_path_database(path='saves/pdg_example_database.pickle', save_fig=False):
    db = pickle.load(open(path, 'rb'))

    for path in db:
        numpy_path = np.array([p.value for p in path.path])
        plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o')
    
    if save_fig:
        plt.savefig("saves/pdg/database.png")
    else:
        plt.show()

def draw_relevant_paths(path="saves/pdg_example_database.pickle", save_fig=False):

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
    
    if save_fig:
        plt.savefig("saves/pdg/relevant_paths.png")
    else:
        plt.show()

def draw_search_tree_steps(path="saves/pdg_example_database.pickle", save_fig=False):
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))

    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = PDG(env, path)

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([15.0, 5.0]))

    pdg.compute_retained_paths(target)

    pdg.init_search(start, target)

    for i in range(5):
        pdg.step_search(i)
        pdg.draw_tree(plt.gca())

        plt.pause(0.1)
    

# draw_path_database()
# draw_relevant_paths()
draw_search_tree_steps()