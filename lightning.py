import numpy as np
import pickle
import time

from database import Database

from space import PointRobot
from circle_approximation import ApproximationSpace
from obstacle_sets import BiasedPassage
import matplotlib.pyplot as plt
from rrt import BiDirectionalRRT, RRT
from path import Path
from matplotlib.collections import LineCollection
from utils import smooth_path, interpolate_path



class Lightning():
    def __init__(self, db_path):
        self.db : Database = pickle.load(open(db_path, 'rb'))
        # self.db.paths = [self.db.paths[i] for i in [1,2,3]]
        self.env = None

    def count_invalid_segments(self, path_validities):
        num_invalid = 0
        prev = True
        for validity in path_validities:
            if prev != validity and validity == False:
                num_invalid += 1
            prev = validity
        return num_invalid


    def compute_candidate_paths(self, start, target, n=5):

        # TODO: SIMPLIFY THIS CODE INTO LIST COMPREHENSION
        # path_starts = []
        # path_ends = []
        # for path in self.db.paths:
        #     path = path.path[:-1]

        #     path_starts.append(path[0].value)
        #     path_ends.append(path[-1].value)
        

        # path_starts = np.array(path_starts)
        # path_ends = np.array(path_ends)

        path_starts = np.array([path[0].value for path in self.db.paths])
        path_targets = np.array([path[-1].value for path in self.db.paths])
        # TODO: SIMPLIFY THIS CODE INTO LIST COMPREHENSION

        start_dists = np.linalg.norm(path_starts - start.value, axis=1)
        target_dists = np.linalg.norm(path_targets - target.value, axis=1)

        summed_dists = start_dists + target_dists
        ordered_path_idxes = np.argsort(summed_dists)

        best_n_path_idxes = ordered_path_idxes[:n]

        retained_paths_invalid_segment_counts = []
        computed_path_validities = []

        # # TODO: Change to one call to batch is valid and split up the results later (similar to batch is valid edge)
        # for path_idx in best_n_path_idxes:
        #     path = self.db.paths[path_idx]
        #     # path = path.path[:-1]

        #     path_states = np.array([state.value for state in path])
        #     validities = self.env.batch_is_valid(path_states)

        #     # print(validities)
        #     # print(f"Num Invalid Segments: {self.count_invalid_segments(validities)}")

        #     # self.env.draw_environment(plt.gca())
        #     # for s in path:
        #     #     self.env.space.draw_state(plt.gca(), s)
            
        #     # plt.show()
        #     # plt.clf()

        #     num_invalid_segments = self.count_invalid_segments(validities)
        #     retained_paths_invalid_segment_counts.append(num_invalid_segments)

        # Updated validation calculations
        path_lengths = [0] + [len(path) for path in self.db.paths]
        path_idxes = np.cumsum(path_lengths)

        all_path_states = np.array([state.value for path in self.db.paths for state in path])

        all_path_states1 = all_path_states[:-1]
        all_path_states2 = all_path_states[1:]

        state_validities = self.env.batch_is_valid(all_path_states)
        edge_validities = self.env.batch_is_valid_edge(all_path_states1, all_path_states2)

        for path_idx in best_n_path_idxes:
            # path = self.db.paths[path_idx]

            start_idx = path_idxes[path_idx]
            end_idx = path_idxes[path_idx+1]

            path_state_validities = state_validities[start_idx:end_idx]
            path_edge_validities = edge_validities[start_idx:end_idx-1]

            # print(path_state_validities, len(path_state_validities))
            # print(path_edge_validities)

            invalid_edges = np.where(path_edge_validities == False)[0]

            path_state_validities[invalid_edges] = False
            path_state_validities[(invalid_edges+1)] = False

            # print(path_state_validities, 'final validities')

            # print(invalid_edges)
            # print(path_state_validities[invalid_edges])
            # exit(0)



            num_invalid_segments = self.count_invalid_segments(path_state_validities)
            retained_paths_invalid_segment_counts.append(num_invalid_segments)
            computed_path_validities.append(path_state_validities)
        
        most_valid_path_idx = np.argmin(retained_paths_invalid_segment_counts)
        # path_idx = best_n_path_idxes[most_valid_path_idx]


        # print(len(self.db.paths[path_idx]), len(computed_path_validities[path_idx]))
        # print()
        # for path_idx in best_n_path_idxes:
        #     print(len(self.db.paths[best_n_path_idxes[path_idx]]), len(computed_path_validities[most_valid_path_idx]))

        path_idx = best_n_path_idxes[most_valid_path_idx]

        # self.draw(plt.gca(), self.db.paths[path_idx])
        # plt.show()
        # plt.clf()

        return path_idx, computed_path_validities[most_valid_path_idx] #, retained_paths_invalid_segment_counts[num_invalid_segments], invalid_segments
    
    def repair_path(self, path, path_validities):
        # print("type in repair path:", type(path))
        path = path.path

        num_invalid = 0
        prev = True
        intervals_start = []
        intervals_end = []
        # print(path_validities, 'here')
        # print(len(path))
        
        for idx, validity in enumerate(path_validities):
            if prev != validity and validity == False:
                num_invalid += 1
                intervals_start.append(idx)
            elif prev == False and validity == True:
                intervals_end.append(idx)
            prev = validity
        
        intervals = list(zip(intervals_start, intervals_end))


        repaired_segments = []
        for seg_start_idx, seg_end_idx in intervals:
            rrt = BiDirectionalRRT(env)
            # print(seg_start_idx, seg_end_idx, len(path))
            repaired_segment = rrt.search(path[seg_start_idx-1], path[seg_end_idx+1], max_steps=10000)
            repaired_segments.append(repaired_segment)
            # print("repaired_segment", len(repaired_segment))

        for i in reversed(range(len(intervals))):
            seg_start_idx, seg_end_idx = intervals[i]
            path = path[0:seg_start_idx] + repaired_segments[i].path + path[seg_end_idx+1:]
        
        
    
        rrt = BiDirectionalRRT(env)
        start_to_path = rrt.search(start, path[0], max_steps=1000)

        rrt = BiDirectionalRRT(env)
        path_to_target = rrt.search(path[-1], target, max_steps=1000)
        
        return start_to_path.path + path + path_to_target.path


    def search(self, env, start, target, n=10):
        start_time = time.time()

        self.env = env
        path_idx, path_validity = self.compute_candidate_paths(start, target, n)
        print(f"Time to compute Candidate Paths: {time.time() - start_time}")
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
    
    def draw(self, ax, path):
        self.env.draw_environment(ax)
        path_states = np.array([state.value for state in path.path])
        ax.scatter(path_states[:, 0], path_states[:, 1], color='red')
        path_edges = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
        ax.add_collection(LineCollection(path_edges, color='red'))
        


if __name__ == '__main__':
    db_path = 'saves/database_v5.pickle'
    seed = np.random.randint(0, 100000)
    # seed = 5093
    print(f"Seed: {seed}")
    np.random.seed(seed)

    env = PointRobot()
    
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=8))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))
    # start, target = env.make_state(np.array([4.5, 4.5])), env.make_state(np.array([16.5, 4.5]))

    start, target = env.space.sample_task()

    print("start", start.value)
    print("target", target.value)



    lightning = Lightning(db_path=db_path)
    # lightning.env = env
    

    # lightning.compute_candidate_paths(start, target)
    # env.space.draw_environment(plt.gca())
    # lightning.db.draw_paths(plt.gca())
    # plt.show()
    # plt.clf()

    path = lightning.search(env, start, target)

    lightning.draw(plt.gca(), path)
    plt.show()

    # path = smooth_path(env, path)
    # path = interpolate_path(path, env, 0.1)
    # env.space.animate_path(path=path)

    rrt = RRT(env)
    path = rrt.search(start, target)





        

