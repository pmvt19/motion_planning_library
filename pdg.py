import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from space import PointRobot, RobotSpace
from obstacle_sets import BiasedPassage
from database import Database
from path import Path
from collections import defaultdict
from circle_approximation import ApproximationSpace
from matplotlib.collections import LineCollection


from rrt import RRT

# class Database():
#     def __init__(self):
#         self.paths = []

    # def save_db(self, save_path):
    #     pickle.dump(self, save_path)


# class PDG():
#     def __init__(self, db, delta=0.5, dist_threshold=0.5):
#         self.db = db
#         self.dist_threshold = dist_threshold

#     def compute_path_metadata(self):
#         self.path_lengths = [0]
#         self.path_states = []
#         self.path_idx_tracker = []

#         for path in self.db.paths:
#             path = path.path[:-1] # TODO: Remove this HACK
#             self.path_lengths.append(len(path))
#             self.path_states.extend([state.value for state in path])

#         self.path_states = np.array(self.path_states)
#         self.path_states_dists = None

#         min_dists = np.min(self.path_states_dists, axis=1)

#         keep_paths_mask = min_dists < self.dist_threshold

#         keep_paths = np.unique(self.path_idx_tracker[keep_paths_mask])

#     def search(self, start, target):
#         self.paths = []

#         for path in self.db.paths:
#             # path = path.path[:-1] # TODO: Remove this HACK
#             self.paths.append(np.array([state.value for state in path]))
        
#         self.path_cum_dists = []
#         for path in self.paths:
#             path_states1 = path[1:]
#             path_states2 = path[:-1]

#             adj_dists = np.linalg.norm(path_states1 - path_states2, axis=1)

#             cum_dists = np.cumsum(adj_dists[::-1])[::-1]
#             cum_dists = np.append(cum_dists, [0])

#             self.path_cum_dists.append(cum_dists)

#         for path in self.paths:
#             pass

class PDG():
    def __init__(self, env, db_path):
        self.db : Database = pickle.load(open(db_path, 'rb'))
        self.db.paths = self.db.paths[:50]
        self.env : RobotSpace = env
    
    def compute_retained_paths(self, target):

        path_lengths = [0] + [len(path) for path in self.db.paths]
        path_idxes = np.cumsum(path_lengths)

        all_path_states = np.array([state.value for path in self.db.paths for state in path])

        state_dists_to_target = np.linalg.norm(all_path_states - target.value, axis=1)

        closest_idxes = np.array([np.argmin(state_dists_to_target[path_idxes[i]:path_idxes[i+1]]) for i in range(len(self.db.paths))])
        closest_idxes_flattened = closest_idxes + path_idxes[:-1]

        threshold = 1.0
        kept_paths_mask = state_dists_to_target[closest_idxes_flattened] < threshold

        retained_paths = []
        for i, path in enumerate(self.db.paths):
            if kept_paths_mask[i]:
                new_path = path[:closest_idxes[i]]
                # print(len(new_path))
                if len(new_path) > 0:
                    retained_paths.append(Path(new_path))

        # print(len(retained_paths), len(kept_paths_mask))

        new_db = Database()
        new_db.paths = retained_paths#[:1]
        new_db.draw_paths(plt.gca())
        self.env.draw_environment(plt.gca())
        plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
        plt.scatter(target.value[0], target.value[1], color='red', s=100)
        plt.show()
        plt.clf()

        # TODO: Need to validate the edge from path to goal

        # Keep valid path segments
        # if pre_validate:
        retained_path_lengths = [0] + [len(path) for path in retained_paths]
        # print(retained_path_lengths)
        retained_path_idxes = np.cumsum(retained_path_lengths)
        all_retained_path_states = np.array([state.value for path in retained_paths for state in path])

        path_state_validities = self.env.batch_is_valid(all_retained_path_states)

        validated_paths = []
        for i, path in enumerate(retained_paths):
            s = retained_path_idxes[i]
            e = retained_path_idxes[i+1]

            invalid_portions = np.where(path_state_validities[s:e] == False)[0]
            if len(invalid_portions) > 0:
                validated_paths.append(Path(path.path[invalid_portions[-1]+1:]))
            else:
                validated_paths.append(Path(path.path))
        
        # new_db = Database()
        # new_db.paths = validated_paths
        # new_db.draw_paths(plt.gca())
        # self.env.draw_environment(plt.gca())
        # plt.scatter(target.value[0], target.value[1], color='red', s=100)
        # plt.show()
        # plt.clf()
        # self.validated_paths = validated_paths
        self.validated_paths = retained_paths
        self.compute_c2g_for_paths(self.validated_paths, target)

    def compute_c2g_for_paths(self, paths, target):
        # Paths require the last node to be the goal

        # dist_to_goal = []
        path_c2gs = []
        for path in paths:
            path = np.array([state.value for state in path.path])
            # path[:-1] - path[1:]
            # state_dist_to_goal = np.linalg.norm(path - target.value, axis=1) # TODO: Computation is incorrect
            state_dist_to_goal = np.linalg.norm(path[:-1] - path[1:], axis=1)
            state_dist_to_goal = np.concatenate((state_dist_to_goal, np.array([0])))
            state_c2g = np.cumsum(state_dist_to_goal[::-1])[::-1]
            # print(state_c2g)
            # exit()
            path_c2gs.append(state_c2g)
        # print([path.path for path in paths])
        # print([len(path.path) for path in paths])
        # print(path_c2gs)

        self.flattened_c2gs = np.array([c2g for path in path_c2gs for c2g in path])


    def search(self, start, target):
        self.start = start
        self.target = target

        self.tree = defaultdict(list)

        self.tree[start]

        new_state = self.env.make_state(np.array([5.5, 5.5]))
        self.tree[start].append(new_state)
        self.tree[new_state]

        rrt = RRT(self.env)
        path = rrt.search(start, target, max_steps=5)
        self.tree = rrt.tree
        
        tree_states = np.array([key.value for key in self.tree])
        path_starting_idxes = np.array([len(path) for path in self.validated_paths])
        path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))
        path_states = np.array([state.value for path in self.validated_paths for state in path])

        # print(tree_states.shape, path_states.shape)

        # path_states = np.array([4, 6]).reshape(1, 2)

        # pairwise dist

        # distance_mat = np.sqrt(np.sum(tree_states**2, axis=2, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T)))
        # print(np.sum(tree_states**2, axis=1, keepdims=True), np.sum(path_states**2, axis=1, keepdims=True), 'here', (tree_states @ path_states.T))
        dist_mat = np.sqrt(np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T)))

        # print(dist_mat)
        # print(tree_states)
        threshold = 1.0
        # threshold = 0.5
        # print(path_states[path_states[:, 0] < 10])
        dist_mat[dist_mat > threshold] = np.inf
        # print(np.min(dist_mat))
        c2g_estimates = dist_mat + self.flattened_c2gs

        # print(self.flattened_c2gs.shape)
        # print(np.min(c2g_estimates))

        # print(c2g_estimates.shape, np.sum(c2g_estimates == np.inf, axis=1))
        # print(c2g_estimates.shape, np.sum(c2g_estimates != np.inf, axis=1))

        # TODO: if c2g  estimates becomes fully np.inf, we need to explore
        potential_connection_edges_idxes = np.where(c2g_estimates != np.inf)

        edge_starts = tree_states[potential_connection_edges_idxes[0]]
        edge_ends = path_states[potential_connection_edges_idxes[1]]

        # print(np.hstack((edge_starts, edge_ends)))
        # print(len(edge_starts))

        start_time = time.time()
        edge_validities = self.env.batch_is_valid_edge(edge_starts, edge_ends)
        end_time = time.time()
        
        # print(edge_validities)
        # print(f"Time to Validate Edges: {end_time - start_time}")

        c2g_estimates[potential_connection_edges_idxes[0][edge_validities == False]] = np.inf
        c2g_estimates[potential_connection_edges_idxes[1][edge_validities == False]] = np.inf

        # print(np.argmin(c2g_estimates))
        # print(np.where(c2g_estimates == np.min(c2g_estimates)))

        # print(np.unravel_index(np.argmin(c2g_estimates), c2g_estimates.shape))
        
        tree_state_idx, path_state_idx = np.unravel_index(np.argmin(c2g_estimates), c2g_estimates.shape)
        # print(path_starting_idxes)
        # print(np.where(path_state_idx < path_starting_idxes))
        path_starting_idx = np.where(path_state_idx < path_starting_idxes)[0][0] - 1
        print("path starting idx", path_starting_idx)

        following_path = path_states[path_state_idx:path_starting_idxes[path_starting_idx+1]]
        start_time = time.time()
        path_state_validities = self.env.batch_is_valid(following_path)
        end_time = time.time()
        print(f"Time to validate following path: {end_time - start_time}")

        invalid_idx = np.where(path_state_validities == False)[0][0]
        print(invalid_idx, np.where(path_state_validities == False)) # TODO: There could be nothing in invalid idx, in which case we just add the entire final path

        deletion_offset = 1
        add_to_tree_segment = following_path[:invalid_idx]
        kept_path_segment = following_path[invalid_idx+deletion_offset:] # TODO: Figure out logic for updating path states and all

        self.validated_paths[path_starting_idx] = kept_path_segment

        print(following_path, path_state_validities)

        parent = self.env.make_state(tree_states[tree_state_idx])
        for state in add_to_tree_segment:
            child = self.env.make_state(state)
            self.tree[parent].append(child)
            self.tree[child]
            parent = child

        # path_starting_idxes[path_state_idx] 
        # path_state_idx + 1

        # tree_node = self.env.make_state(tree_states[tree_state_idx])
        # new_child_node = self.env.make_state(path_states[path_state_idx])

        # self.tree[tree_node].append(new_child_node)
        # self.tree[new_child_node]

        print("Adding:", path_states[path_state_idx])

        self.draw_tree(plt.gca())

        plt.gca().scatter(following_path[:, 0], following_path[:, 1], color='orange', zorder=0, s=100)

        plt.show()

        # print(potential_connection_edges_idxes)

        # print(self.flattened_c2gs)
        # flattened_c2gs = []

        # find closest on each path

        # find node with lowest c2g estimate and attach to the associated path (follow until collision)


    def draw_tree(self, ax):
        self.env.draw_environment(ax)
        nodes = np.array([node.value for node in self.tree.keys()])
        ax.scatter(nodes[:, 0], nodes[:, 1], color='pink')
        ax.scatter(self.start.value[0], self.start.value[1], s=100, c='green')
        ax.scatter(self.target.value[0], self.target.value[1], s=100, c='red')

        edges = [[(p.value[0], p.value[1]), (c.value[0], c.value[1])] for p in self.tree for c in self.tree[p]]
        edges = LineCollection(edges, color='blue')
        ax.add_collection(edges)

if __name__ == '__main__':
    np.random.seed(0)

    # db_save_path = 'saves/database.pickle'
    # db = pickle.load(open(db_save_path, 'rb'))


    # path_lengths = [0]
    # path_states = []

    # for path in db.paths:
    #     path = path.path[:-1]
    #     path_lengths.append(len(path))
    #     path_states.extend([state.value for state in path])

    # path_states = np.array(path_states)

    db_save_path = 'saves/database_v4.pickle'
    
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=8))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = PDG(env, db_save_path)

    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))
    start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([45.0, 5.0]))

    start_time = time.time()
    pdg.compute_retained_paths(target)
    end_time = time.time()
    print(f"Time to compute paths: {end_time-start_time}")

    pdg.search(start, target)

    # pdg.search(start, target)
    
    # print(path_lengths, np.cumsum(path_lengths))

    # print(path_states.shape)

    # print(np.hstack((path_states, np.arange(352).reshape(-1, 1))).tolist())



    

    # # np.random.seed(0)
    # env = PointRobot()
    # env.set_obstacles(BiasedPassage(bias=0.5, num_walls=1))

    # # start, target = env.sample_task()
    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))

    # rrt = RRT(env=env, delta=0.1)
    # path = rrt.search(start, target, goal_bias=0.1)
    
    # rrt.draw_tree(plt.gca(), path=path)
    # plt.show()





    