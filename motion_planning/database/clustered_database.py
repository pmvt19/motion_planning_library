from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from matplotlib.collections import LineCollection

from motion_planning.database import Database
from motion_planning.tools import Path


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
            rp_numpy_path = self._path_to_numpy_path(
                                self[path_idx_representative_for_cluster_id]
                            )
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
    
    def merge_dbs(self, other_db: Database):
        if self.clusters is None:
            super().merge_dbs(other_db)
        else:
            # Options:
            # 1. Remove the clusters
            # 2. Add Each Path Individually to Preserve Clusters

            # This function implements Option 2
            for path in other_db.paths:
                self.add_path(path)

    def subsample_database(self, num_paths_per_cluster: int = 30) -> Database:
        db = Database()
        
        kept_paths: list[Path] = []
        for cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            cluster_size = len(cluster)
            kept_cluster_paths_idxes = np.random.randint(low=0, 
                                                         high=cluster_size, 
                                                         size=(min(cluster_size, 
                                                                   num_paths_per_cluster),))

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
                ax.scatter(
                    path_states[:, 0], 
                    path_states[:, 1],
                    color=cmap(cluster_id),
                    zorder=2
                )
                path_edges = [
                    (path[j].value[:2], path[j+1].value[:2])
                    for j in range(len(path)-1)
                ]
                ax.add_collection(LineCollection(path_edges, color=cmap(cluster_id)))

    def draw_cluster(self, ax, cluster_id, color='blue'):
        for _, path_idx in enumerate(self.clusters[cluster_id]):
            path = self[path_idx]
            path_states = np.array([state.value for state in path.path])
            ax.scatter(
                path_states[:, 0], path_states[:, 1], color=color, zorder=2
            )
            path_edges = [
                (path[j].value[:2], path[j+1].value[:2])
                for j in range(len(path)-1)
            ]
            ax.add_collection(LineCollection(path_edges, color=color))