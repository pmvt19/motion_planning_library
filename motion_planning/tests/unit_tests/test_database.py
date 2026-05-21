import random
import unittest

from motion_planning.database import Database
from motion_planning.tools import Path


class TestDatabase(unittest.TestCase):
    def test_adding_path_to_database(self):
        self.db = Database()

        original_database_length = len(self.db)

        path = Path([])  # Create an Empty Path
        self.db.add_path(path)

        new_database_length = len(self.db)

        # Original Database should be 1 element smaller than the new database size
        self.assertEqual(original_database_length + 1, new_database_length)

    def test_database_only_accepts_path_objects(self):
        self.db = Database()

        # Cannot add a non Path object to the database
        with self.assertRaises(AssertionError):
            self.db.add_path([])

    def test_database_returns_correct_length(self):
        self.db = Database()

        # Randomly generate number of paths to create
        num_paths = random.randint(1, 500)

        for _ in range(num_paths):
            self.db.add_path(Path([]))
        
        self.assertEqual(num_paths, len(self.db))

    def test_database_indexing(self):
        self.db = Database()

        path0 = Path([])
        path1 = Path([])

        self.db.add_path(path0)
        self.db.add_path(path1)

        # Check DB Indexing refers to correct path objects
        self.assertTrue(self.db[0] is path0)
        self.assertTrue(self.db[1] is path1)

        # Check DB Indexing does not refer to correct incorrect path objects
        self.assertFalse(self.db[0] is path1)
        self.assertFalse(self.db[1] is path0)

    def test_batch_adding_paths_to_database(self):
        self.db = Database()

        original_db_size = len(self.db)

        batch_num_paths = random.randint(1, 500)
        batch_of_paths = [Path([]) for _ in range(batch_num_paths)]

        self.db.batch_add_paths(batch_of_paths)

        self.assertEqual(original_db_size + batch_num_paths, len(self.db))
