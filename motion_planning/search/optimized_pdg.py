import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from collections import defaultdict
from matplotlib.collections import LineCollection
from sklearn.neighbors import KDTree

from motion_planning.space import PointRobot, RobotSpace
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.database import Database
from motion_planning.path import Path
from motion_planning.circle_approximation import ApproximationSpace
from motion_planning.utils import interpolate_path, smooth_path
from motion_planning.rrt import RRT

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

        # dist_to_goal = []
        path_c2gs = []
        for path in paths:
            path = np.array([state.value for state in path.path])
            # path[:-1] - path[1:]
            # state_dist_to_goal = np.linalg.norm(path - target.value, axis=1) # TODO: Computation is incorrect
            # print(path.shape)
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


    def search(self, start, target):

        timing_dict = defaultdict(list)

        self.start = start
        self.target = target

        self.tree = defaultdict(list)


        self.tree[start]
        self.child_to_parent = {}
        self.child_to_parent[start] = None
        
        do_rrt = False

        tree_states = np.array([key.value for key in self.tree])
        path_starting_idxes = np.array([len(path) for path in self.validated_paths])
        path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))
        path_states = np.array([state.value for path in self.validated_paths for state in path])

        path_state_path_idxes = np.array([i for i, path in enumerate(self.validated_paths) for _ in path])
        c2g_estimates = self.compute_c2g_estimates_for_states(tree_states, path_states)
        for i in range(5000):
            start_time = time.time()
            
            path_starting_idxes = np.array([len(path) for path in self.validated_paths])
            path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))
            # print(path_starting_idxes)
            # path_states = np.array([state.value for path in self.validated_paths for state in path])
            end_time = time.time()
            timing_dict['flattening_states_time'].append(end_time - start_time)

            # pairwise dist

            # start_time = time.time()
            # dist_mat = np.sqrt(np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T)))

            
            # threshold = 1.0
            # threshold = 5.0
            # threshold = 0.5
            
            # dist_mat[dist_mat > threshold] = np.inf
            # dist_mat[dist_mat == 0.0] = np.inf
            
            # c2g_estimates = dist_mat + self.flattened_c2gs
            # end_time = time.time()
            # timing_dict['dist_c2g_calc_time'].append(end_time - start_time)
            
            # min_c2g_estimate = np.min(c2g_estimates) if len(c2g_estimates) > 0 else np.inf
            min_c2g_estimate = np.inf
            if c2g_estimates.shape[1] > 0:
                # print(c2g_estimates.shape)
                min_c2g_estimate = np.min(c2g_estimates)
            expansion_tech = None
            
            if min_c2g_estimate == np.inf or do_rrt:
                # Do RRT for a couple of steps


                rrt = RRT(self.env, delta=2)
                path_rrt = rrt.search(start, target, max_steps=10, starting_tree_info=(self.tree,self.child_to_parent))
                # path_rrt = rrt.search(start, target, max_steps=1000, starting_tree_info=(self.tree,self.child_to_parent))
                self.tree = rrt.tree
                self.child_to_parent = rrt.child_to_parent


                # print("RRTing")
                expansion_tech = 'rrt'
                do_rrt = False

                # exp_node, sampled_point = self.select_node(goal_bias=0)
                # cur_node = self.expand_node(exp_node, sampled_point)
            else:
                # print("PDGing")
                expansion_tech = 'pdg'

                potential_connection_edges_idxes = np.where(c2g_estimates != np.inf)

                vals = c2g_estimates[potential_connection_edges_idxes[0], potential_connection_edges_idxes[1]]

                # num_connections_to_keep = 1500
                num_connections_to_keep = 4500
                # num_connections_to_keep = 9000

                argsorted_vals = np.argsort(vals)

                # best_n_connections = np.argsort(vals)[:num_connections_to_keep]
                # other_n_connections = np.argsort(vals)[num_connections_to_keep:]

                best_n_connections = argsorted_vals[:num_connections_to_keep]
                other_n_connections = argsorted_vals[num_connections_to_keep:]

                edge_starts = tree_states[potential_connection_edges_idxes[0][best_n_connections]]
                edge_ends = path_states[potential_connection_edges_idxes[1][best_n_connections]]

                start_time = time.time()
                edge_validities = self.env.batch_is_valid_edge_uniform(edge_starts, edge_ends)
                end_time = time.time()

                timing_dict['validate_potential_connection_time'].append(end_time - start_time)


                c2g_estimates[(potential_connection_edges_idxes[0][best_n_connections][edge_validities == False]), (potential_connection_edges_idxes[1][best_n_connections][edge_validities == False])] = np.inf
                c2g_estimates[(potential_connection_edges_idxes[0][other_n_connections]), (potential_connection_edges_idxes[1][other_n_connections])] = np.inf

                if np.min(c2g_estimates) == np.inf:
                    do_rrt = True
                    continue
                
                tree_state_idx, path_state_idx = np.unravel_index(np.argmin(c2g_estimates), c2g_estimates.shape)

                path_starting_idx = np.where(path_state_idx < path_starting_idxes)[0][0] - 1

                following_path = path_states[path_state_idx:path_starting_idxes[path_starting_idx+1]]
                start_time = time.time()
                path_state_validities = self.env.batch_is_valid(following_path)
                

                if len(following_path) == 1:
                    path_edge_validities = []
                else: 
                    start_edge_states = following_path[:-1]
                    end_edge_states = following_path[1:]

                    path_edge_validities = self.env.batch_is_valid_edge(start_edge_states, end_edge_states)
                    end_time = time.time()
                    timing_dict['validate_follow_path'].append(end_time - start_time)

                path_edge_validities = np.concatenate(([True], path_edge_validities))
                path_validity = np.logical_and(path_state_validities, path_edge_validities)


                start_time = time.time()
                invalid_idxes = np.where(path_validity == False)[0]
                if len(invalid_idxes) == 0:
                    add_to_tree_segment = following_path
                    kept_path_segment = following_path

                    updated_path_states = path_states
                    updated_flattened_c2gs = self.flattened_c2gs
                    updated_path_state_path_idxes = path_state_path_idxes
                    updated_c2g_estimates = c2g_estimates

                else:
                    invalid_idx = invalid_idxes[0]
                    deletion_offset = 1
                    add_to_tree_segment = following_path[:invalid_idx]
                    si = path_starting_idxes[path_starting_idx]

                    updated_path_states = np.vstack((path_states[:si], path_states[path_state_idx+invalid_idx+deletion_offset:]))
                    updated_path_state_path_idxes = np.hstack((path_state_path_idxes[:si], path_state_path_idxes[path_state_idx+invalid_idx+deletion_offset:]))
                    updated_flattened_c2gs = np.concatenate((self.flattened_c2gs[:si], self.flattened_c2gs[path_state_idx+invalid_idx+deletion_offset:]), axis=0)

                    updated_c2g_estimates = np.concatenate((c2g_estimates[:, :si], c2g_estimates[:, path_state_idx+invalid_idx+deletion_offset:]), axis=1)     
    
                    kept_path_segment = following_path[invalid_idx+deletion_offset:] # TODO: Figure out logic for updating path states and all

                    self.validated_paths[path_starting_idx] = Path([self.env.make_state(state) for state in kept_path_segment])

                    

                    if self.use_delete_radius:
                        bad_state_in_follow_path = following_path[invalid_idx]
                        kd_tree = KDTree(data=updated_path_states)
                        idxes = kd_tree.query_radius(bad_state_in_follow_path.reshape(1, -1), r=2.0)[0]
                        flattened_idxes = idxes.flatten()
                        
                        path_idxes_to_modify = updated_path_state_path_idxes[idxes].flatten()

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
                            path_starting_idxes = np.array([len(p) for p in self.validated_paths])
                            path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))

                            # Find Starting Idx of Path
                            my_si = path_starting_idxes[cur_path_idx]

                            my_kept_path_segment = updated_path_states[(state_to_delete+deletion_offset):path_starting_idxes[cur_path_idx+1]]

                            # Delete all the states up until and including the deletion state
                            updated_path_states = np.vstack((updated_path_states[:my_si], updated_path_states[state_to_delete+deletion_offset:]))
                            updated_path_state_path_idxes = np.hstack((updated_path_state_path_idxes[:my_si], updated_path_state_path_idxes[state_to_delete+deletion_offset:]))

                            updated_flattened_c2gs = np.concatenate((updated_flattened_c2gs[:my_si], updated_flattened_c2gs[state_to_delete+deletion_offset:]), axis=0)
                            updated_c2g_estimates = np.concatenate((updated_c2g_estimates[:, :my_si], updated_c2g_estimates[:, state_to_delete+deletion_offset:]), axis=1)

                            # Update Path in self.validated_paths
                            self.validated_paths[cur_path_idx] = Path([self.env.make_state(state) for state in my_kept_path_segment])

                            
        
                end_time = time.time()
                timing_dict['filter_following_path'].append(end_time - start_time)

                start_time = time.time()
                parent = self.env.make_state(tree_states[tree_state_idx])
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
                end_time = time.time()
                timing_dict['add_states_to_tree'].append(end_time - start_time)

                added_states = np.array(added_states)
                # print(tree_states.shape, added_states.reshape(-1, 2).shape)
                tree_states = np.vstack((tree_states, added_states.reshape(-1, 2))) # TODO HACK BUG HARDCODED!!!


                
                

                




                # timing_dict['add_states_to_tree'].append(end_time - start_time)
                
                start_time = time.time()
                path_states = updated_path_states
                path_state_path_idxes = updated_path_state_path_idxes
                self.flattened_c2gs = updated_flattened_c2gs
                
                
                end_time = time.time()
                timing_dict['recompute_cost_to_gos'].append(end_time - start_time)


                start_time = time.time()
                # print(added_states.shape, updated_path_states.shape)
                new_states_c2g_estimates = self.compute_c2g_estimates_for_states(added_states, updated_path_states)
                updated_c2g_estimates_full = np.concatenate((updated_c2g_estimates, new_states_c2g_estimates), axis=0)
                c2g_estimates = updated_c2g_estimates_full
                end_time = time.time()
                timing_dict['dist_c2g_calc_time'].append(end_time - start_time)
            
            if target in self.tree:
                print(f"Found Path in {i} iterations")
                self.timing_dict = timing_dict
                return self.backtrack(target)
            
            
            # self.compute_c2g_for_paths(self.validated_paths, target)
            # path_states = updated_path_states
            # self.flattened_c2gs = updated_flattened_c2gs
            # end_time = time.time()
            # timing_dict['recompute_cost_to_gos'].append(end_time - start_time)

            self.timing_dict = timing_dict
        print(f"Iteration: {i}")

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

if __name__ == '__main__':
    seed = np.random.randint(0, 10000)
    # seed = 575
    # seed = 8458

    # seed = 7277
    # seed = 1610

    # seed = 4132
    # seed = 9221
    # seed = 7594
    # seed = 3032
    # seed = 7272
    # seed = 4567 # USE FOR OPTIMIZATION TESTING

    # seed = 5896 # DEBUG SEED

    # seed = 2735
    # seed = 2480
    # seed = 3411 # DEBUG DONE
    # seed = 3501 # DEBUG HANGING (Maybe Fixed)
    # seed = 228 # Debug (Empty path) (Done)

    # seed = 3184
    # seed = 4092
    # seed = 4501

    # Broken Seeds (on PC)
    # seed = 4459
    # seed = 9264 # Broken on Mac (Fixed?)
    # seed = 8718 # BUG SEED

    # Optimization Seed
    # seed = 4054 #IMportant

    # seed = 1119
    # seed = 6473

    # seed = 3967 # Slow Search
    # seed = 6292
    # seed = 5185
    # seed = 6300 # Very slow search
    print(f"Using Seed: {seed}")
    np.random.seed(seed)

    # db_save_path = 'saves/database.pickle'
    # db = pickle.load(open(db_save_path, 'rb'))


    # path_lengths = [0]
    # path_states = []

    # for path in db.paths:
    #     path = path.path[:-1]
    #     path_lengths.append(len(path))
    #     path_states.extend([state.value for state in path])

    # path_states = np.array(path_states)

    # db_save_path = 'saves/database_v5.pickle'
    db_save_path = "saves/database_bpe3_large.pickle"
    # db_save_path = 'saves/database_v1_bpe3.pickle'
    
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = OptimizedPDG(env, db_save_path)

    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))
    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([45.0, 5.0]))
    start, target = env.space.sample_task()
    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([35.0, 5.0]))

    start_time = time.time()
    pdg.compute_retained_paths(target)
    end_time = time.time()
    print(f"Time to compute paths: {end_time-start_time}")

    start_time = time.time()
    path = pdg.search(start, target)
    end_time = time.time()
    print(f"Time to search: {end_time - start_time}")

    for key in pdg.timing_dict:
        print(f"{key}: {np.sum(pdg.timing_dict[key])}")
        # print(pdg.timing_dict[key])

    pdg.draw_tree(plt.gca(), path)
    plt.show()

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
