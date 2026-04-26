import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from collections import defaultdict
from matplotlib.collections import LineCollection
from sklearn.neighbors import KDTree

from motion_planning.space import PointRobot, RobotSpace
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.database import Database, ClusteredDatabase
from motion_planning.tools import Path
from motion_planning.space import ApproximationSpace
from motion_planning.utils import interpolate_path, smooth_path
from motion_planning.search import RRT

class OptimizedPDG():
    def __init__(self, env, db_path):
        self.db : Database = pickle.load(open(db_path, 'rb'))
        # self.db.paths = self.db.paths[:50]
        self.env : RobotSpace = env

        self.use_delete_radius = True
    
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
                if len(new_path) > 0:
                    retained_paths.append(Path(path=new_path + [target]))

        # new_db = Database()
        # new_db.paths = retained_paths#[:1]
        # new_db.draw_paths(plt.gca())
        # self.env.draw_environment(plt.gca())
        # plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
        # plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)

        # plt.show()
        # plt.clf()

        # TODO: Need to validate the edge from path to goal

        # Keep valid path segments
        # if pre_validate:
        # retained_path_lengths = [0] + [len(path) for path in retained_paths]
        # # print(retained_path_lengths)
        # retained_path_idxes = np.cumsum(retained_path_lengths)
        # all_retained_path_states = np.array([state.value for path in retained_paths for state in path])

        # path_state_validities = self.env.batch_is_valid(all_retained_path_states)

        # validated_paths = []
        # for i, path in enumerate(retained_paths):
        #     s = retained_path_idxes[i]
        #     e = retained_path_idxes[i+1]

        #     invalid_portions = np.where(path_state_validities[s:e] == False)[0]
        #     if len(invalid_portions) > 0:
        #         validated_paths.append(Path(path.path[invalid_portions[-1]+1:]))
        #     else:
        #         validated_paths.append(Path(path.path))
        
        # new_db = Database()
        # new_db.paths = validated_paths
        # new_db.draw_paths(plt.gca())
        # self.env.draw_environment(plt.gca())
        # plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
        # plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
        # plt.show()
        # plt.clf()
        # self.validated_paths = validated_paths
        self.validated_paths = retained_paths
        self.compute_c2g_for_paths(self.validated_paths, target)

    def compute_c2g_for_paths(self, paths, target):
        # Paths require the last node to be the goal

        path_c2gs = []
        for path in paths:
            path = np.array([state.value for state in path.path])
            state_dist_to_goal = np.linalg.norm(path[:-1] - path[1:], axis=1)
            state_dist_to_goal = np.concatenate((state_dist_to_goal, np.array([0])))
            state_c2g = np.cumsum(state_dist_to_goal[::-1])[::-1]
            path_c2gs.append(state_c2g)

        self.flattened_c2gs = np.array([c2g for path in path_c2gs for c2g in path])

    def optimized_compute_c2g_for_paths(self):
        pass

    def compute_c2g_estimates_for_states(self, tree_states, path_states):
        if tree_states.shape[0] == 0:
            return np.empty((0, len(path_states)))
        dist_mat = np.sqrt(np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T)))

        threshold = 5.0
        dist_mat[dist_mat > threshold] = np.inf
        dist_mat[dist_mat == 0.0] = np.inf

        c2g_estimates = dist_mat + self.flattened_c2gs

        return c2g_estimates
    
    def get_tree_path_state_info(self):
        path_starting_idxes = np.array([len(path) for path in self.validated_paths])
        self.path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))
        return self.path_starting_idxes
    
    def filter_connection_attempts(self, c2g_estimates, tree_states, path_states):
        potential_connection_edges_idxes = np.where(c2g_estimates != np.inf)

        vals = c2g_estimates[potential_connection_edges_idxes[0], potential_connection_edges_idxes[1]]

        num_connections_to_keep = 4500 # TODO: Make Configuration Variable

        argsorted_vals = np.argsort(vals)

        best_n_connections = argsorted_vals[:num_connections_to_keep]
        other_n_connections = argsorted_vals[num_connections_to_keep:]

        edge_starts = tree_states[potential_connection_edges_idxes[0][best_n_connections]]
        edge_ends = path_states[potential_connection_edges_idxes[1][best_n_connections]]

        edge_validities = self.env.batch_is_valid_edge_uniform(edge_starts, edge_ends)

        c2g_estimates[(potential_connection_edges_idxes[0][best_n_connections][edge_validities == False]), (potential_connection_edges_idxes[1][best_n_connections][edge_validities == False])] = np.inf
        c2g_estimates[(potential_connection_edges_idxes[0][other_n_connections]), (potential_connection_edges_idxes[1][other_n_connections])] = np.inf
        return c2g_estimates

    def get_follow_path(self, c2g_estimates, path_starting_idxes, path_states):
        tree_state_idx, path_state_idx = np.unravel_index(np.argmin(c2g_estimates), c2g_estimates.shape)
        path_starting_idx = np.where(path_state_idx < path_starting_idxes)[0][0] - 1
        following_path = path_states[path_state_idx:path_starting_idxes[path_starting_idx+1]]
        return tree_state_idx, following_path, path_starting_idx, path_state_idx
    
    def validate_follow_path(self, following_path):
        path_state_validities = self.env.batch_is_valid(following_path)

        if len(following_path) == 1:
            path_edge_validities = []
        else: 
            start_edge_states = following_path[:-1]
            end_edge_states = following_path[1:]

            path_edge_validities = self.env.batch_is_valid_edge(start_edge_states, end_edge_states)

        path_edge_validities = np.concatenate(([True], path_edge_validities))
        path_validity = np.logical_and(path_state_validities, path_edge_validities)
        return path_validity
    
    def filter_invalid_segments_of_follow_path(self, following_path, path_validity, path_starting_idx, path_state_idx):
        invalid_idxes = np.where(path_validity == False)[0]
        if len(invalid_idxes) == 0:
            add_to_tree_segment = following_path
            kept_path_segment = following_path

        else:
            invalid_idx = invalid_idxes[0]
            deletion_offset = 1
            add_to_tree_segment = following_path[:invalid_idx]
            si = self.path_starting_idxes[path_starting_idx]

            self.path_states = np.vstack((self.path_states[:si], self.path_states[path_state_idx+invalid_idx+deletion_offset:]))
            self.path_state_path_idxes = np.hstack((self.path_state_path_idxes[:si], self.path_state_path_idxes[path_state_idx+invalid_idx+deletion_offset:]))
            self.flattened_c2gs = np.concatenate((self.flattened_c2gs[:si], self.flattened_c2gs[path_state_idx+invalid_idx+deletion_offset:]), axis=0)

            self.c2g_estimates = np.concatenate((self.c2g_estimates[:, :si], self.c2g_estimates[:, path_state_idx+invalid_idx+deletion_offset:]), axis=1)     

            kept_path_segment = following_path[invalid_idx+deletion_offset:] # TODO: Figure out logic for updating path states and all

            self.validated_paths[path_starting_idx] = Path([self.env.make_state(state) for state in kept_path_segment])


            if self.use_delete_radius:
                bad_state_in_follow_path = following_path[invalid_idx]
                kd_tree = KDTree(data=self.path_states)
                idxes = kd_tree.query_radius(bad_state_in_follow_path.reshape(1, -1), r=2.0)[0]
                flattened_idxes = idxes.flatten()
                
                path_idxes_to_modify = self.path_state_path_idxes[idxes].flatten()

                unique_path_idxes_to_modify = np.unique(path_idxes_to_modify)

                deletion_states = []
                for path_idx in unique_path_idxes_to_modify:
                    idx_to_delete = np.max(flattened_idxes[path_idxes_to_modify == path_idx])
                    deletion_states.append(idx_to_delete)
                
                deletion_states = np.array(deletion_states)
                deletion_states_and_path_idxes = np.hstack((deletion_states.reshape(-1, 1), unique_path_idxes_to_modify.reshape(-1, 1)))

                argsorted_deletion_states_and_path_idxes = np.argsort(deletion_states_and_path_idxes[:, 0])[::-1]
                sorted_deletion_states_and_path_idxes = deletion_states_and_path_idxes[argsorted_deletion_states_and_path_idxes]

                for state_to_delete, cur_path_idx in sorted_deletion_states_and_path_idxes:
                    self.path_starting_idxes = self.get_tree_path_state_info()

                    # Find Starting Idx of Path
                    my_si = self.path_starting_idxes[cur_path_idx]

                    my_kept_path_segment = self.path_states[(state_to_delete+deletion_offset):self.path_starting_idxes[cur_path_idx+1]]

                    # Delete all the states up until and including the deletion state
                    self.path_states = np.vstack((self.path_states[:my_si], self.path_states[state_to_delete+deletion_offset:]))
                    self.path_state_path_idxes = np.hstack((self.path_state_path_idxes[:my_si], self.path_state_path_idxes[state_to_delete+deletion_offset:]))

                    self.flattened_c2gs = np.concatenate((self.flattened_c2gs[:my_si], self.flattened_c2gs[state_to_delete+deletion_offset:]), axis=0)
                    self.c2g_estimates = np.concatenate((self.c2g_estimates[:, :my_si], self.c2g_estimates[:, state_to_delete+deletion_offset:]), axis=1)

                    # Update Path in self.validated_paths
                    self.validated_paths[cur_path_idx] = Path([self.env.make_state(state) for state in my_kept_path_segment])
        return add_to_tree_segment, kept_path_segment
    
    def add_state_from_path_to_tree(self, parent, add_to_tree_segment):
        added_states = []
        for state in add_to_tree_segment:
            if np.all(parent.value == state): # HACK: This might be a hack (may need to figure out why this is happening)
                continue
            
            child = self.env.make_state(state)
            if child in self.tree:
                break

            self.tree[parent].append(child)
            self.tree[child]
            added_states.append(state)

            self.child_to_parent[child] = parent
            parent = child

        added_states = np.array(added_states)
        self.tree_states = np.vstack((self.tree_states, added_states.reshape(-1, 2))) # TODO HACK BUG HARDCODED!!!
    
    def step_search(self, iteration):
        self.path_starting_idxes = self.get_tree_path_state_info()

        min_c2g_estimate = np.inf
        if self.c2g_estimates.shape[1] > 0:
            min_c2g_estimate = np.min(self.c2g_estimates)

        if min_c2g_estimate == np.inf or self.do_rrt:
            rrt = RRT(self.env, delta=2)
            _ = rrt.search(self.start, self.target, max_steps=10, starting_tree_info=(self.tree, self.child_to_parent))

            self.tree = rrt.tree
            self.child_to_parent = rrt.child_to_parent

            self.do_rrt = False
        else:
            self.c2g_estimates = self.filter_connection_attempts(self.c2g_estimates, self.tree_states, self.path_states)

            if np.min(self.c2g_estimates) == np.inf:
                self.do_rrt = True
                return
            
            tree_state_idx, following_path, path_starting_idx, path_state_idx = self.get_follow_path(self.c2g_estimates, self.path_starting_idxes, self.path_states)

            path_validity = self.validate_follow_path(following_path)
            add_to_tree_segment, kept_path_segment = self.filter_invalid_segments_of_follow_path(following_path, path_validity, path_starting_idx, path_state_idx)

            parent_node = self.env.make_state(self.tree_states[tree_state_idx])
            self.add_state_from_path_to_tree(parent_node, add_to_tree_segment)
        
        if self.target in self.tree:
            print(f"Found Path in {iteration} iterations")
            return

    def init_search(self, start, target):
        self.start = start
        self.target = target

        self.tree = defaultdict(list)

        self.tree[self.start]
        self.child_to_parent = {}

        self.child_to_parent[self.start] = None
        self.do_rrt = False

        # Optimized PDG Specific Initializations
        self.tree_states = np.array([key.value for key in self.tree])
        self.path_states = np.array([state.value for path in self.validated_paths for state in path]).reshape(-1, 2) # TODO: Use env dims for the "2" arg
        self.path_state_path_idxes = np.array([i for i, path in enumerate(self.validated_paths) for _ in path])
        self.c2g_estimates = self.compute_c2g_estimates_for_states(self.tree_states, self.path_states)

    def search(self, start, target, max_steps=5000):
        self.init_search(start, target)

        for i in range(max_steps):
            self.step_search(i)

            if self.target in self.tree:
                return self.backtrack(self.target)

        return Path([])

    def backtrack(self, end=None):
        if end is None or end not in self.child_to_parent:
            return Path()

        path = []
        node = end
        while node:
            path.append(node)
            node = self.child_to_parent[node]
        return Path(path=path[::-1])

    def draw_tree(self, ax, path=None, show_task=True):
        self.env.draw_environment(ax)
        nodes = np.array([node.value for node in self.tree.keys()])
        ax.scatter(nodes[:, 0], nodes[:, 1], color='pink')

        edges = [[(p.value[0], p.value[1]), (c.value[0], c.value[1])] for p in self.tree for c in self.tree[p]]
        edges = LineCollection(edges, color='blue')
        ax.add_collection(edges)

        if show_task:
            ax.scatter(self.start.value[0], self.start.value[1], s=100, c='green')
            ax.scatter(self.target.value[0], self.target.value[1], s=100, c='red')

        if path:
            path = [(path[i].value[:2], path[i+1].value[:2]) for i in range(len(path)-1)]
            ax.add_collection(LineCollection(path, color='red'))
        
    def select_node(self, goal_bias=0):
        if np.random.random() < goal_bias:
            sampled_point = self.target
        else:
            sampled_point = self.env.sample_valid_point()
        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point

    def expand_node(self, node, sampled_point):
        self.delta = 0.5
        if np.linalg.norm(self.target.value - node.value) < self.delta:
            new_node = self.target
        else:
            # print(np.linalg.norm(sampled_point.value - node.value), 'dist')
            # new_node = self.env.shoot_ray(node, sampled_point, self.delta)
            new_node = self.env.shoot_ray(node, sampled_point, min(self.delta, np.linalg.norm(sampled_point.value - node.value)))
            # assert(new_node.value[1] < 10), "ERROR EXPAND NODE"
        if new_node != node:
            self.tree[node].append(new_node)
            self.tree[new_node]
            self.child_to_parent[new_node] = node

        return new_node

class BiDirectionalPDG():
    def __init__(self, env, db_path):
        self.env = env
        self.db_path = db_path
        self.max_connection_distance = 0.5

        start_time = time.time()
        self.forward_pdg = OptimizedPDG(self.env, self.db_path)
        self.backward_pdg = OptimizedPDG(self.env, self.db_path)
        end_time = time.time()
        print(f"Time to Instantiate PDGs: {end_time - start_time}")

    def attempt_tree_connection(self, forward_pdg, backward_pdg):
        forward_tree_nodes = np.array([node.value for node in forward_pdg.tree.keys()])
        backward_tree_nodes = np.array([node.value for node in backward_pdg.tree.keys()])

        kdt = KDTree(forward_tree_nodes)
        dist, ind = kdt.query(backward_tree_nodes, k=1)
        dist = dist.flatten()
        ind = ind.flatten()

        if np.min(dist) < self.max_connection_distance:
            backward_tree_node_idx = np.argmin(dist)
            forward_tree_node_idx = ind[backward_tree_node_idx]
            return self.env.is_valid_edge(self.env.make_state(forward_tree_nodes[forward_tree_node_idx]), self.env.make_state(backward_tree_nodes[backward_tree_node_idx])), \
                    (self.env.make_state(forward_tree_nodes[forward_tree_node_idx]), self.env.make_state(backward_tree_nodes[backward_tree_node_idx]))
        return False, None
    
    def delete_duplicate_states_in_path(self, path):
        i = len(path) - 1
        while i > 0:
            if np.all(path[i-1] == path[i]):
                path.pop(i)
            i -= 1
        return path

    def backtrack(self, forward_pdg: OptimizedPDG, backward_pdg: OptimizedPDG, connection):
        start_time = time.time()
        if connection is None:
            return Path()
        forward_end_state, backward_end_state = connection
        forward_path = forward_pdg.backtrack(end=forward_end_state)
        backward_path = backward_pdg.backtrack(end=backward_end_state)

        # Join Both Paths and Reverse the backward PDG Tree path
        joined_path = forward_path.path + backward_path.path[::-1] 
        joined_path = self.delete_duplicate_states_in_path(joined_path)
        end_time = time.time()
        print(f"Time to backtrack path: {end_time - start_time}")
        return Path(path=joined_path)

    def search(self, start, target, max_steps=500):
        self.start = start
        self.target = target

        start_time = time.time()
        self.forward_pdg.compute_retained_paths(self.target)
        self.backward_pdg.compute_retained_paths(self.start)
        end_time = time.time()
        print(f"Time to Compute Retained Paths: {end_time - start_time}")

        start_time = time.time()
        self.forward_pdg.init_search(start, target)
        self.backward_pdg.init_search(target, start)
        end_time = time.time()
        print(f"Time to Initialize Search: {end_time - start_time}")

        start_time = time.time()
        for i in range(max_steps):
            self.forward_pdg.step_search(i)
            self.backward_pdg.step_search(i)
            is_connected, connection = self.attempt_tree_connection(self.forward_pdg, self.backward_pdg)

            if is_connected:
                break
        end_time = time.time()
        print(f"Time to search (without computing retained paths): {end_time - start_time}")
        
        if is_connected:
            return self.backtrack(self.forward_pdg, self.backward_pdg, connection)
        else:
            return Path([])
        
    def draw_tree(self, ax, path=None, show_task=True):
        self.env.draw_environment(ax)
        self.backward_pdg.draw_tree(ax=ax, path=None, show_task=False) 
        self.forward_pdg.draw_tree(ax=ax, path=path, show_task=show_task)
        

if __name__ == '__main__':
    seed = np.random.randint(0, 10000)
    # seed = 5700
    # seed = 4349
    print(f"Using Seed: {seed}")
    np.random.seed(seed)

    # db_save_path = 'saves/database_v5.pickle'
    # db_save_path = "saves/database_bpe3_large.pickle"
    # db_save_path = "saves/database_bpe8_large.pickle"
    # db_save_path = 'saves/database_v1_bpe3.pickle'
    # db_save_path = 'saves/clustered_database_large_bpe_subsampled.pickle'
    # db_save_path = 'saves/clustered_database_large_bpe_mp_sampler.pickle'
    # db_save_path = 'saves/clustered_database_large_bpe_subsampled.pickle'
    db_save_path = 'saves/smoothed_interpolated_database_large_bpe_subsampled.pickle'
    
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = OptimizedPDG(env, db_save_path)

    start, target = env.space.sample_task()
    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([85.0, 5.0]))

    start_time = time.time()
    pdg.compute_retained_paths(target)
    end_time = time.time()
    time_to_compute_retained_paths = end_time-start_time
    print(f"Time to compute paths: {time_to_compute_retained_paths}")

    start_time = time.time()
    path = pdg.search(start, target)
    end_time = time.time()
    time_to_search = end_time - start_time
    print(f"Time to search: {time_to_search}")

    print(f"Full Optimized PDG Planning Time: {time_to_search + time_to_compute_retained_paths}")

    pdg.draw_tree(plt.gca(), path)
    plt.show()

    bidirectional_pdg = BiDirectionalPDG(env, db_save_path)

    start_time = time.time()
    path = bidirectional_pdg.search(start, target)
    end_time = time.time()
    time_to_search = end_time - start_time
    print(f"Time to search (BiDir): {time_to_search}")

    plt.cla()
    bidirectional_pdg.draw_tree(plt.gca(), path)
    plt.show()
