import numpy as np
import pickle
import time

from database import Database

from space import RobotSpace, PointRobot
from circle_approximation import ApproximationSpace
from obstacle_sets import BiasedPassage
import matplotlib.pyplot as plt
from rrt import BiDirectionalRRT, RRT
from path import Path
from matplotlib.collections import LineCollection
from utils import smooth_path, interpolate_path



class Lightning():
    def __init__(self, env, db_path, max_repair_steps=10000):
        self.db : Database = pickle.load(open(db_path, 'rb'))
        self.env : RobotSpace = env
        self.max_repair_steps : int = max_repair_steps

    def count_invalid_segments(self, path_validities):
        num_invalid = 0
        prev = True
        for validity in path_validities:
            if prev != validity and validity == False:
                num_invalid += 1
            prev = validity
        return num_invalid


    def compute_candidate_paths(self, start, target, n=5):

        path_starts = np.array([path[0].value for path in self.db.paths])
        path_targets = np.array([path[-1].value for path in self.db.paths])

        start_dists = np.linalg.norm(path_starts - start.value, axis=1)
        target_dists = np.linalg.norm(path_targets - target.value, axis=1)

        summed_dists = start_dists + target_dists
        ordered_path_idxes = np.argsort(summed_dists)

        best_n_path_idxes = ordered_path_idxes[:n]

        retained_paths_invalid_segment_counts = []
        computed_path_validities = []

        path_lengths = [0] + [len(path) for path in self.db.paths]
        path_idxes = np.cumsum(path_lengths)

        all_path_states = np.array([state.value for path in self.db.paths for state in path])

        all_path_states1 = all_path_states[:-1]
        all_path_states2 = all_path_states[1:]

        state_validities = self.env.batch_is_valid(all_path_states)
        edge_validities = self.env.batch_is_valid_edge(all_path_states1, all_path_states2)

        for path_idx in best_n_path_idxes:

            start_idx = path_idxes[path_idx]
            end_idx = path_idxes[path_idx+1]

            path_state_validities = state_validities[start_idx:end_idx]
            path_edge_validities = edge_validities[start_idx:end_idx-1]

            invalid_edges = np.where(path_edge_validities == False)[0]

            path_state_validities[invalid_edges] = False
            path_state_validities[(invalid_edges+1)] = False

            num_invalid_segments = self.count_invalid_segments(path_state_validities)
            retained_paths_invalid_segment_counts.append(num_invalid_segments)
            computed_path_validities.append(path_state_validities)
        
        most_valid_path_idx = np.argmin(retained_paths_invalid_segment_counts)

        path_idx = best_n_path_idxes[most_valid_path_idx]

        return path_idx, computed_path_validities[most_valid_path_idx]
    
    def repair_path(self, path, path_validities):
        path = path.path

        num_invalid = 0
        prev = True
        intervals_start = []
        intervals_end = []
        
        for idx, validity in enumerate(path_validities):
            if prev != validity and validity == False:
                num_invalid += 1
                intervals_start.append(idx)
            elif prev == False and validity == True:
                intervals_end.append(idx)
            prev = validity
        
        intervals = list(zip(intervals_start, intervals_end))

        repaired_segments = []
        rrt = RRT(self.env)

        for seg_start_idx, seg_end_idx in intervals:

            repaired_segment = rrt.search(path[seg_start_idx-1], path[seg_end_idx], max_steps=self.max_repair_steps)
            repaired_segments.append(repaired_segment)

        for i in reversed(range(len(intervals))):
            seg_start_idx, seg_end_idx = intervals[i]
            path = path[0:seg_start_idx] + repaired_segments[i].path + path[seg_end_idx+1:]
    
        start_to_path = rrt.search(self.start, path[0], max_steps=self.max_repair_steps)
        path_to_target = rrt.search(path[-1], self.target, max_steps=self.max_repair_steps)
        
        return start_to_path.path + path + path_to_target.path


    def search(self, start, target, n=10):
        start_time = time.time()
        self.start = start
        self.target = target

        path_idx, path_validity = self.compute_candidate_paths(start, target, n)
        print(f"Time to compute Candidate Paths: {time.time() - start_time}")
        self.path_idx = path_idx
        self.path_validity = path_validity
        # Get Validities
        path = self.db.paths[path_idx]
        # path_states = np.array([state.value for state in path])
        # st = time.time()
        # validities = self.env.batch_is_valid(path_states)
        # et = time.time()
        # print(f"Time to validate final candidate path: {et-st}")

        repair_start_time = time.time()
        repaired_path = self.repair_path(path, path_validity)
        end_time = time.time()

        print(f"Time to repair path: {end_time - repair_start_time}")

        print(f"Lightning Total Search Time: {end_time - start_time}")

        return Path(path=repaired_path)
    
    def _compute_interval_segments(self):
        prev = True
        intervals_segs = [0]
        self.path_validity[35:37] = False
        for idx, validity in enumerate(self.path_validity):
            if prev != validity and validity == False:
                intervals_segs.append(idx)
            elif prev == False and validity == True:
                intervals_segs.append(idx)
            prev = validity
        intervals_segs.append(len(self.path_validity))
        return intervals_segs
        
    
    def draw(self, ax, path, show_task=True, show_unrepaired_path=True, verbose=True):
        self.env.draw_environment(ax)
        path_states = np.array([state.value for state in path.path])
        ax.scatter(path_states[:, 0], path_states[:, 1])
        path_edges = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
        ax.add_collection(LineCollection(path_edges, color='red'))

        if show_task:
            ax.scatter(self.start.value[0], self.start.value[1], s=100, c='green')
            ax.scatter(self.target.value[0], self.target.value[1], s=100, c='red')

        if show_unrepaired_path and not verbose:
            unrepaired_path = self.db.paths[self.path_idx]
            unrepaired_path_states = np.array([state.value for state in unrepaired_path.path])

            ax.scatter(unrepaired_path_states[:, 0], unrepaired_path_states[:, 1], color='orange')
            unrepaired_path_edges = [(unrepaired_path[i].value[:2], unrepaired_path[i+1].value[:2]) for i in range(len(unrepaired_path)-1)]
            ax.add_collection(LineCollection(unrepaired_path_edges, color='purple'))

        elif show_unrepaired_path and verbose:
            unrepaired_path = self.db.paths[self.path_idx]
            unrepaired_path_states = np.array([state.value for state in unrepaired_path.path])

            intervals_segs = self._compute_interval_segments()
            for i in range(len(intervals_segs)-1):
                start_idx = intervals_segs[i]
                end_idx = intervals_segs[i+1]

                if self.path_validity[start_idx]:
                    alpha_val = 1
                    edge_zorder = 2
                    unrepaired_path_edges = [(unrepaired_path[i].value[:2], unrepaired_path[i+1].value[:2]) for i in range(start_idx, end_idx-1)]
                else:
                    unrepaired_path_edges = [(unrepaired_path[i].value[:2], unrepaired_path[i+1].value[:2]) for i in range(start_idx-1, end_idx)]
                    alpha_val = 0.8
                    edge_zorder = 0

                ax.scatter(unrepaired_path_states[start_idx:end_idx, 0], unrepaired_path_states[start_idx:end_idx, 1], color='orange', alpha=alpha_val)
                ax.add_collection(LineCollection(unrepaired_path_edges, color='purple', alpha=alpha_val, zorder=edge_zorder))
        


if __name__ == '__main__':
    db_path = 'saves/database_v4.pickle'
    # db_path = 'saves/database_v1_bpe3.pickle'
    seed = np.random.randint(0, 100000)
    # seed = 5093
    # seed = 75809
    # seed = 61606
    # seed = 9222
    seed = 19695
    print(f"Seed: {seed}")
    np.random.seed(seed)

    env = PointRobot()
    
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=8))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    start, target = env.space.sample_task()

    print("start", start.value)
    print("target", target.value)



    lightning = Lightning(env=env, db_path=db_path)
    
    # env.space.draw_environment(plt.gca())
    # lightning.db.draw_paths(plt.gca())
    # plt.show()
    # plt.clf()

    path = lightning.search(start, target)

    lightning.draw(plt.gca(), path)
    plt.show()

    # path = smooth_path(env, path)
    # path = interpolate_path(path, env, 0.1)
    # env.space.animate_path(path=path)

    rrt = RRT(env)
    # rrt = BiDirectionalRRT(env)
    path = rrt.search(start, target, max_steps=10000)

    print(f"Total Final Collision Checks: {env.num_collision_checks}", env.space.num_collision_checks)





        

