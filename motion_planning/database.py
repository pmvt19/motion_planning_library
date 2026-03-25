import time
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing
import concurrent.futures

from matplotlib.collections import LineCollection

from motion_planning.prm import IncrementalPRM, PRM
from motion_planning.obstacle_sets import BiasedPassage, RandomSamplePassage
from motion_planning.space import PointRobot
from motion_planning.circle_approximation import ApproximationSpace
from motion_planning.path import Path
from motion_planning.mp_sampler import MPSampler

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
            path_edges = [(path[j].value[:2], path[j+1].value[:2]) for j in range(len(path)-1)]
            ax.add_collection(LineCollection(path_edges, color=cmap(i)))
    
    def add_path(self, path):
        assert(isinstance(path, Path))
        self.paths.append(path)

    def merge_dbs(self, other_db):
        self.paths = self.paths + other_db.paths
    
    def __len__(self):
        return len(self.paths)

    def __add__(self, other_db):
        self.merge_dbs(other_db)
        return self
    
    def __getitem__(self, idx):
        return self.paths[idx]
    
    def populate_db(self, mp_sampler: MPSampler, num_envs: int, num_tasks_per_env: int):
        for i in range(num_envs):
            env = mp_sampler.sample_env()
            prm = IncrementalPRM(env, num_samples=1000, num_neighbors=5)
            prm.create_graph()

            for j in range(num_tasks_per_env):
                print(f"Env {i+1}, Task {j+1}")
                start, target = mp_sampler.sample_task(env)

                path = prm.search(start, target)

                if path:
                    self.add_path(path)
            print(f"DB Size: {len(db)}")
    
def merge_db_lists(dbs):
    db = Database()
    for other_db in dbs:
        db = db + other_db
    return db

def generate_database(db_save_path=None):
    print("WARNING: Hard Coded")
    db = Database()

    for i in range(20):
        env = PointRobot()
        env.set_obstacles(BiasedPassage(num_walls=3, bias=0.5))
        env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

        prm = IncrementalPRM(env, num_samples=1000, num_neighbors=5)
        prm.create_graph()

        for j in range(20):
            print(f"Env {i+1}, Task {j+1}")
            start, target = env.sample_valid_point(), env.sample_valid_point()

            path = prm.search(start, target)
            if path:
                db.paths.append(path)
        print(f"DB Size: {len(db)}")

    if db_save_path:
        print(f"Saving Database to {db_save_path}")
        db.save_to_path(db_save_path)
    return db

def generate_database_parallel(db_save_path=None):
    dbs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(generate_database) for _ in range(10)]

        for f in concurrent.futures.as_completed(results):
            dbs.append(f.result())
    
    return merge_db_lists(dbs)

def generate_database_mp_sampler_version(db_save_path=None):
    mp_sampler = MPSampler(PointRobot(), BiasedPassage, {"num_walls": 1, "bias": 0.5})
    db = Database()

    for i in range(10):
        env = mp_sampler.sample_env()
        prm = IncrementalPRM(env, num_samples=1000, num_neighbors=5)
        prm.create_graph()

        for j in range(10):
            print(f"Env {i+1}, Task {j+1}")
            start, target = mp_sampler.sample_task(env)

            path = prm.search(start, target)

            if path:
                db.paths.append(path)
        print(f"DB Size: {len(db)}")

    if db_save_path:
        print(f"Saving Database to {db_save_path}")
        db.save_to_path(db_save_path)

    return db

if __name__ == '__main__':

    # db_save_path = 'saves/database_v6.pickle'
    # db_save_path = 'saves/database_v1_bpe3.pickle'
    # db_save_path = 'saves/database_bpe3_large.pickle'
    # db_save_path = 'saves/database_bpe3_small.pickle'
    db_save_path = 'saves/database_bpe_mp_sampler.pickle'
    # new_db = generate_database_parallel()

    # new_db = generate_database()
    # new_db.save_to_path(db_save_path)
    # end_time = time.time()
    # print(f"Database Size: {len(new_db)}")
    # print(f"Time to create database: {end_time-start_time}")

    db = Database()
    mp_sampler = MPSampler(PointRobot(), BiasedPassage, {"num_walls": 3, "bias": 0.5})

    db.populate_db(mp_sampler, num_envs=10, num_tasks_per_env=10)
    db.draw_paths(plt.gca())
    plt.show()