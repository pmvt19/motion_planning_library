import unittest

import numpy as np
import matplotlib.pyplot as plt

from motion_planning.tools import NumpyState
from motion_planning.space import RobotSpace, PointRobot, ApproximationSpace
from motion_planning.search import RRT, BiDirectionalRRT, RRTStar, MedialAxisRRT, PRM, LazyPRM, IncrementalPRM, NonUniformPRM, MedialAxisPRM
from motion_planning.obstacle_sets import BiasedPassage


class TestRRTSearchMethods(unittest.TestCase):

    def setUp(self):
        # Create the Environment
        self.env = PointRobot()
        self.env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))
        
        # Manually Define the Task
        self.start, self.target = self.env.make_state(np.array([5.0, 5.0])), self.env.make_state(np.array([15.0, 5.0]))

    def test_rrt(self):
        rrt = RRT(self.env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        rrt.draw_tree(plt.gca(), path, show_task=True)
        plt.show()

    def test_bidirectional_rrt(self):
        rrt = BiDirectionalRRT(self.env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        rrt.draw_tree(plt.gca(), path, show_task=True)
        plt.show()
    
    def test_rrt_star(self):
        rrt = RRTStar(self.env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        rrt.draw_tree(plt.gca(), path, show_task=True)
        plt.show()

## --- RRT Tests --- ##
def test_rrt(env: RobotSpace, start: NumpyState, target: NumpyState):
    rrt = RRT(env)
    path = rrt.search(start, target, max_steps=1000)

    # Visualize Output
    rrt.draw_tree(plt.gca(), path, show_task=True)
    plt.show()

def test_bidirectional_rrt(env: RobotSpace, start: NumpyState, target: NumpyState):
    rrt = BiDirectionalRRT(env)
    path = rrt.search(start, target, max_steps=1000)

    # Visualize Output
    rrt.draw_tree(plt.gca(), path, show_task=True)
    plt.show()

def test_rrt_start(env: RobotSpace, start: NumpyState, target: NumpyState):
    rrt = RRTStar(env)
    path = rrt.search(start, target, max_steps=1000)

    # Visualize Output
    rrt.draw_tree(plt.gca(), path, show_task=True)
    plt.show()

def test_medial_axis_rrt(env: RobotSpace, start: NumpyState, target: NumpyState):
    env = ApproximationSpace(env, do_overapproximation=True)
    rrt = MedialAxisRRT(env)
    path = rrt.search(start, target, max_steps=1000)

    # Visualize Output
    rrt.draw_tree(plt.gca(), path, show_task=True)
    plt.show()

## --- PRM Tests --- ##
def test_prm(env: RobotSpace, start: NumpyState, target: NumpyState):
    prm = PRM(env=env, num_samples=5000, num_neighbors=10, validate_edges=True)
    prm.create_graph()

    path = prm.search(start, target)

    # Visualize Output
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

def test_lazy_prm(env: RobotSpace, start: NumpyState, target: NumpyState):
    prm = LazyPRM(env=env, num_samples=5000, num_neighbors=10)
    prm.create_graph()

    path = prm.search(start, target)

    # Visualize Output
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

def test_incremental_prm(env: RobotSpace, start: NumpyState, target: NumpyState):
    prm = IncrementalPRM(env=env, num_samples=1000, num_neighbors=5)
    prm.create_graph()

    path = prm.search(start, target)

    # Visualize Output
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

def test_nonuniform_prm(env: RobotSpace, start: NumpyState, target: NumpyState):
    prm = IncrementalPRM(env=env, num_samples=1000, num_neighbors=5)
    prm.create_graph()

    path = prm.search(start, target)

    # Visualize Output
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

def test_medial_axis_prm(env: RobotSpace, start: NumpyState, target: NumpyState):
    env = ApproximationSpace(env, do_overapproximation=True)
    prm = MedialAxisPRM(env=env, num_samples=1000, num_neighbors=5)
    prm.create_graph()

    path = prm.search(start, target)

    # Visualize Output
    env.draw_environment(plt.gca())
    prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
    plt.show()

## --- Database Tests --- ##

def test_lightning(env: RobotSpace, start: NumpyState, target: NumpyState):
    pass

def test_pdg(env: RobotSpace, start: NumpyState, target: NumpyState):
    pass

def test_bidirectional_pdg(env: RobotSpace, start: NumpyState, target: NumpyState):
    pass

if __name__ == '__main__':
    # Create the Environment
    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))
    
    # Manually Define the Task
    start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))

    ## Run Visual Search Tests

    # RRTs
    test_rrt(env, start, target)
    test_bidirectional_rrt(env, start, target)
    test_rrt_start(env, start, target)
    test_medial_axis_rrt(env, start, target)

    # PRMs
    test_prm(env, start, target)
    test_lazy_prm(env, start, target)
    test_incremental_prm(env, start, target)
    test_nonuniform_prm(env, start, target)
    test_medial_axis_prm(env, start, target)

    # Database Methods
    test_lightning(env, start, target)
    test_pdg(env, start, target)
    test_bidirectional_pdg(env, start, target)
