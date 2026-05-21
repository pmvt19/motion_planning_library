import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from motion_planning.tools import Path


class Database:
    def __init__(self):
        self.paths = []

    def set_env(self, env):
        self.env = env

    def save_to_path(self, path):
        pickle.dump(self, open(path, "wb"))

    def draw_paths(self, ax):
        cmap = plt.get_cmap("tab10", len(self.paths))
        for i, path in enumerate(self.paths):
            path_states = np.array([state.value for state in path.path])
            ax.scatter(path_states[:, 0], path_states[:, 1], color=cmap(i), zorder=2)
            path_edges = [
                (path[j].value[:2], path[j + 1].value[:2]) for j in range(len(path) - 1)
            ]
            ax.add_collection(LineCollection(path_edges, color=cmap(i)))

    def add_path(self, path):
        assert isinstance(path, Path)
        path_idx = len(self.paths)
        self.paths.append(path)
        return path_idx

    def batch_add_paths(self, paths: list[Path]):
        self.paths.extend(paths)

    def merge_dbs(self, other_db: "Database"):
        self.paths = self.paths + other_db.paths

    def __len__(self):
        return len(self.paths)

    def __add__(self, other_db):
        self.merge_dbs(other_db)
        return self

    def __getitem__(self, idx):
        return self.paths[idx]

    @staticmethod
    def load_db(db_save_path: str) -> "Database":
        return pickle.load(open(db_save_path, "rb"))
