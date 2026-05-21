import os
import unittest

from motion_planning.database import Database, ClusteredDatabase
from motion_planning.database.db_annotator import populate_db
from motion_planning.experiments.utils.mp_sampler import MPSampler
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import BiasedPassage


class TestClusteredDatabase(unittest.TestCase):
    def setUp(self):
        self.db: ClusteredDatabase = self.create_or_load_clustered_database()
    
    def create_or_load_clustered_database(self) -> ClusteredDatabase:
        self.db_path = "saves/tests/test_clustered_database_db.pickle"

        if not os.path.exists(self.db_path):
            db = ClusteredDatabase()
            mp_sampler = MPSampler(
                PointRobot(), BiasedPassage, {"num_walls": 1, "bias": 0.5}
            )
            populate_db(
                db, mp_sampler, num_envs=5, num_tasks_per_env=5, smooth_paths=False
            )

            # Cluster the database and print the metadata
            db.cluster(threshold=250)
            db.print_cluster_info()

            # Save DB to local directory for future use in tests
            db.save_to_path(self.db_path)

            # Ensure Created Database is Non-Empty
            self.assertGreater(len(db), 0)
        else:
            db = ClusteredDatabase.load_db(self.db_path)
    
        return db
    
    def create_database(self) -> Database:
        db = Database()
        mp_sampler = MPSampler(
            PointRobot(), BiasedPassage, {"num_walls": 1, "bias": 0.5}
        )
        populate_db(
            db, mp_sampler, num_envs=2, num_tasks_per_env=2, smooth_paths=False
        )
        return db

    def test_merge_databases_with_clusters(self):
        other_db = self.create_database()
        for path in other_db:
            self.db.add_path(path)
        
        num_clustered_paths = sum([len(cluster) for cluster in self.db.clusters.values()])
        self.assertEqual(num_clustered_paths, len(self.db))

    def test_clustered_database_subsampling(self):
        num_paths_per_cluster = 2
        num_clusters = len(self.db.clusters)

        subsampled_db = self.db.subsample_database(num_paths_per_cluster=num_paths_per_cluster)

        self.assertLessEqual(len(subsampled_db), num_paths_per_cluster*num_clusters)

    def test_erase_clustering(self):
        self.db.erase_clustering()
        self.assertTrue(self.db.clusters is None)
        self.assertTrue(self.db.clustered_threshold is None)