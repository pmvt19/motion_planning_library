import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.search import (
    PRM,
    RRT,
    BiDirectionalRRT,
    IncrementalPRM,
    LazyPRM,
    MedialAxisPRM,
    MedialAxisRRT,
    NonUniformPRM,
    RRTStar,
    Lightning,
    OptimizedPDG,
    OptimizedBiDirectionalPDG
)
from motion_planning.space import ApproximationSpace, PointRobot, RobotSpace
from motion_planning.database import Database
from motion_planning.database.db_annotator import populate_db
from motion_planning.experiments.utils.mp_sampler import MPSampler

class TestRRTSearchMethods(unittest.TestCase):
    def setUp(self):
        # Create the Environment
        self.env: RobotSpace = PointRobot()
        self.env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))

        # Manually Define the Task
        self.start = self.env.make_state(np.array([5.0, 5.0]))
        self.target = self.env.make_state(np.array([15.0, 5.0]))

    def visualize_search(self, rrt: RRT, path):
        rrt.draw_tree(plt.gca(), path, show_task=True)
        plt.show()

    def test_rrt(self):
        rrt = RRT(self.env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        self.visualize_search(rrt, path)

    def test_bidirectional_rrt(self):
        rrt = BiDirectionalRRT(self.env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        self.visualize_search(rrt, path)

    def test_rrt_star(self):
        rrt = RRTStar(self.env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        self.visualize_search(rrt, path)

    def test_medial_axis_rrt(self):
        local_env = ApproximationSpace(self.env, do_overapproximation=True)
        rrt = MedialAxisRRT(local_env)
        path = rrt.search(self.start, self.target, max_steps=1000)

        # Visualize Output
        self.visualize_search(rrt, path)


class TestPRMSearchMethods(unittest.TestCase):
    def setUp(self):
        # Create the Environment
        self.env = PointRobot()
        self.env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))

        # Manually Define the Task
        self.start = self.env.make_state(np.array([5.0, 5.0]))
        self.target = self.env.make_state(np.array([15.0, 5.0]))
    
    def visualize_search(self, prm: PRM, path):
        self.env.draw_environment(plt.gca())
        prm.draw(plt.gca(), path=path, show_task=True, plot_invalid_edges=False)
        plt.show()

    def test_prm(self):
        prm = PRM(env=self.env, num_samples=5000, num_neighbors=10, validate_edges=True)
        prm.create_graph()

        path = prm.search(self.start, self.target)

        self.visualize_search(prm, path)

    def test_lazy_prm(self):
        prm = LazyPRM(env=self.env, num_samples=5000, num_neighbors=10)
        prm.create_graph()

        path = prm.search(self.start, self.target)

        self.visualize_search(prm, path)

    def test_incremental_prm(self):
        prm = IncrementalPRM(env=self.env, num_samples=1000, num_neighbors=5)
        prm.create_graph()

        path = prm.search(self.start, self.target)

        self.visualize_search(prm, path)

    def test_nonuniform_prm(self):
        prm = NonUniformPRM(env=self.env, num_samples=1000, num_neighbors=5)
        prm.create_graph()

        path = prm.search(self.start, self.target)

        self.visualize_search(prm, path)

    def test_medial_axis_prm(self):
        local_env = ApproximationSpace(self.env, do_overapproximation=True)
        prm = MedialAxisPRM(env=local_env, num_samples=1000, num_neighbors=5)
        prm.create_graph()

        path = prm.search(self.start, self.target)

        self.visualize_search(prm, path)


class TestDatabaseSearchMethods(unittest.TestCase):

    def setUp(self):
        self.create_database()

        # Create the Environment
        self.env = PointRobot()
        self.env.set_obstacles(BiasedPassage(num_walls=1, bias=0.5))

        # Manually Define the Task
        self.start = self.env.make_state(np.array([5.0, 5.0]))
        self.target = self.env.make_state(np.array([15.0, 5.0]))

    def create_database(self):
        self.db_path = "saves/tests/test_database_search_method_db.pickle"

        if not os.path.exists(self.db_path):
            db = Database()
            mp_sampler = MPSampler(PointRobot(), BiasedPassage, {"num_walls": 3, "bias": 0.5})
            populate_db(db, mp_sampler, num_envs=5, num_tasks_per_env=10, smooth_paths=False)
            db.save_to_path(self.db_path)

            # Ensure Created Database is Non-Empty
            self.assertGreater(len(db), 0)

    def test_lightning(self):
        lightning = Lightning(env=self.env, db_path=self.db_path)
        path = lightning.search(self.start, self.target)

        self.env.draw_environment(plt.gca())
        lightning.draw(plt.gca(), path=path, show_task=True)
        plt.show()

    def test_pdg(self):
        pdg = OptimizedPDG(env=self.env, db_path=self.db_path)
        pdg.compute_retained_paths(self.target)
        path = pdg.search(self.start, self.target)

        self.env.draw_environment(plt.gca())
        pdg.draw_tree(plt.gca(), path=path)
        plt.show()

    def test_bidirectional_pdg(self):
        pdg = OptimizedBiDirectionalPDG(env=self.env, db_path=self.db_path)
        path = pdg.search(self.start, self.target)

        self.env.draw_environment(plt.gca())
        pdg.draw_tree(plt.gca(), path=path)
        plt.show()
