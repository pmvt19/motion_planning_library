import time
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing
import concurrent.futures

from fastdtw import fastdtw
from collections import defaultdict
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
        path_idx = len(self.paths)
        self.paths.append(path)
        return path_idx
    
    def batch_add_paths(self, paths: list[Path]):
        self.paths.extend(paths)

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
            print(f"DB Size: {len(self)}")
    
    @staticmethod
    def load_db(db_save_path: str) -> "Database":
        return pickle.load(open(db_save_path, "rb"))

class ClusteredDatabase(Database):
    def __init__(self):
        super().__init__()

        self.clusters = None
        self.clustered_threshold = None

    def _path_to_numpy_path(self, path):
        states = []
        for state in path:
            states.append(state.value)
        return states

    def cluster_single_path(self, path_idx):
        path = self[path_idx]
        numpy_path = self._path_to_numpy_path(path)

        dists = []
        for cluster_id in self.clusters:
            path_idx_representative_for_cluster_id = self.clusters[cluster_id][0]
            rp_numpy_path = self._path_to_numpy_path(self[path_idx_representative_for_cluster_id])
            dtw_distance, _ = fastdtw(numpy_path, rp_numpy_path, dist=2)

            dists.append(dtw_distance)
        
        best_cluster_id = np.argmin(dists)
        if dists[best_cluster_id] < self.clustered_threshold:
            self.clusters[best_cluster_id].append(path_idx)
        else:
            self.clusters[len(self.clusters)].append(path_idx)

    def cluster(self, threshold=10):
        self.clustered_threshold = threshold

        self.clusters = defaultdict(list)
        self.clusters[0] = [0]

        for i, path in enumerate(self.paths):
            print(f"Clustering Path: {i}/{len(self.paths)}", end='\r')
            # Skip the first path since it must belong to cluster 0
            if i == 0:
                continue

            self.cluster_single_path(i)
        print(f"Num Clusters Generated: {len(self.clusters)}")

    def erase_clustering(self):
        self.clusters = None
        self.clustered_threshold = None

    def add_path(self, path):
        path_idx = super().add_path(path)
        if self.clusters is not None:
            self.cluster_single_path(path_idx)
    
    def merge_dbs(self, other_db):
        if self.clusters is None:
            super().merge_dbs(other_db)
        else:
            # Options:
            # 1. Remove the clusters
            # 2. Add Each Path Individually to Preserve Clusters
            # raise NotImplementedError
            for path in other_db.paths:
                self.add_path(path)

    def subsample_database(self, num_paths_per_cluster: int = 30) -> Database:
        db = Database()
        
        kept_paths: list[Path] = []
        for cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            cluster_size = len(cluster)
            kept_cluster_paths_idxes = np.random.randint(0, cluster_size, size=(min(cluster_size, num_paths_per_cluster),))

            for kept_path_idx in kept_cluster_paths_idxes:
                path_idx = cluster[kept_path_idx]
                kept_paths.append(self[path_idx])
        
        db.batch_add_paths(kept_paths)
        return db
    
    def print_cluster_info(self):
        for cluster_id in self.clusters:
            print(f"Cluster: {cluster_id}, Size: {len(self.clusters[cluster_id])}")

    def draw_clusters(self, ax):
        for cluster_id in self.clusters:
            cmap = plt.get_cmap('tab10', len(self.clusters))
            for _, path_idx in enumerate(self.clusters[cluster_id]):
                path = self[path_idx]
                path_states = np.array([state.value for state in path.path])
                ax.scatter(path_states[:, 0], path_states[:, 1], color=cmap(cluster_id), zorder=2)
                path_edges = [(path[j].value[:2], path[j+1].value[:2]) for j in range(len(path)-1)]
                ax.add_collection(LineCollection(path_edges, color=cmap(cluster_id)))

    def draw_cluster(self, ax, cluster_id, color='blue'):
        for _, path_idx in enumerate(self.clusters[cluster_id]):
            path = self[path_idx]
            path_states = np.array([state.value for state in path.path])
            ax.scatter(path_states[:, 0], path_states[:, 1], color=color, zorder=2)
            path_edges = [(path[j].value[:2], path[j+1].value[:2]) for j in range(len(path)-1)]
            ax.add_collection(LineCollection(path_edges, color=color))

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
    # db_save_path = 'saves/database_bpe_mp_sampler.pickle'
    # db_save_path = 'saves/clustered_database_bpe_mp_sampler.pickle'
    db_save_path = 'saves/clustered_database_large_bpe_mp_sampler.pickle'
    # new_db = generate_database_parallel()

    # new_db = generate_database()
    # new_db.save_to_path(db_save_path)
    # end_time = time.time()
    # print(f"Database Size: {len(new_db)}")
    # print(f"Time to create database: {end_time-start_time}")

    # db = Database()
    db = ClusteredDatabase()
    mp_sampler = MPSampler(PointRobot(), BiasedPassage, {"num_walls": 3, "bias": 0.5})

    # db.populate_db(mp_sampler, num_envs=30, num_tasks_per_env=20)
    # # db.draw_paths(plt.gca())
    # # plt.show()

    # db.save_to_path(db_save_path)
    # exit()

    db = pickle.load(open(db_save_path, "rb"))

    # db.cluster(threshold=250)
    # db.save_to_path(db_save_path)

    # db.draw_clusters(plt.gca())
    env = PointRobot()
    env.set_obstacles(BiasedPassage(num_walls=3, bias=0.5))
    
    # for cluster_id in db.clusters:
    #     plt.cla()
    #     env.draw_environment(plt.gca())
    #     db.draw_cluster(plt.gca(), cluster_id)
    #     plt.show()
    
    db.print_cluster_info()

    print(f"Size of Clustered DB: {len(db)}")
    ss_db = db.subsample_database(5)
    print(f"Size of Subsampled DB: {len(ss_db)}")

    # ss_db.draw_paths(plt.gca())
    # plt.show()

    ss_db.save_to_path("saves/clustered_database_large_bpe_subsampled.pickle")

    