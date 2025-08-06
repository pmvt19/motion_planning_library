import pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection

from prm import IncrementalPRM, PRM
from obstacle_sets import BiasedPassage
from space import PointRobot
from circle_approximation import ApproximationSpace

class Database():
    def __init__(self):
        self.paths = []
    
    def set_env(self, env):
        self.env = env
    
    def save_to_path(self, path):
        pickle.dump(self, open(path, 'wb'))

    def draw_paths(self, ax):
        cmap = plt.get_cmap('tab10', len(self.paths))
        for i, path in enumerate(self.paths):
            path_states = np.array([state.value for state in path.path])
            ax.scatter(path_states[:, 0], path_states[:, 1], color=cmap(i), zorder=2)
            path_edges = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
            ax.add_collection(LineCollection(path_edges, color=cmap(i)))
    
    def add_path(self, path):
        raise NotImplementedError

    def merge_dbs(self, other_db):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.paths)

    def __add__(self, other_db):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def populate_db(self, env, num_envs, num_paths_per_env):
        raise NotImplementedError
    

if __name__ == '__main__':

    # db_save_path = 'saves/database_v6.pickle'
    db_save_path = 'saves/database_v1_bpe3.pickle'

    db = Database()

    for i in range(20):
        env = PointRobot()
        env.set_obstacles(BiasedPassage(num_walls=3, bias=0.5))
        env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

        # prm = IncrementalPRM(env, num_samples=1000, num_neighbors=20)
        prm = PRM(env, num_samples=5000, num_neighbors=10)
        prm.create_graph()

        for j in range(20):
            print(f"Env {i+1}, Task {j+1}")
            start, target = env.sample_valid_point(), env.sample_valid_point()

            path = prm.search(start, target)
            if path:
                db.paths.append(path)
        print(f"DB Size: {len(db)}")


    db.save_to_path(db_save_path)