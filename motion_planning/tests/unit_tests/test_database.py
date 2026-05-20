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
        pass

    def test_database_returns_correct_length(self):
        pass

    def test_database_indexing(self):
        pass

    def test_batch_adding_paths_to_database(self):
        pass
