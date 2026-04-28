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
from motion_planning.tools import Path
from motion_planning.space import ApproximationSpace
from motion_planning.utils import interpolate_path, smooth_path
from motion_planning.search import RRT

class PDG():
    def __init__(self, env, db_path, prevalidate_paths=False):
        self.db: Database = pickle.load(open(db_path, 'rb'))
        # self.db.paths = self.db.paths[:50]
        self.env: RobotSpace = env
        self.prevalidate_paths: bool = prevalidate_paths
    
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
                # new_path = interpolate_path(new_path, self.env, 0.1).path
                # new_path = smooth_path(self.env, Path(new_path)).path
                # print(len(new_path))
                if len(new_path) > 0:
                    retained_paths.append(Path(path=new_path + [target]))

        # print(len(retained_paths), len(kept_paths_mask))

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
        retained_path_lengths = [0] + [len(path) for path in retained_paths]
        # print(retained_path_lengths)
        retained_path_idxes = np.cumsum(retained_path_lengths)
        all_retained_path_states = np.array([state.value for path in retained_paths for state in path])

        path_state_validities = self.env.batch_is_valid(all_retained_path_states)

        self.validated_paths = retained_paths

        if self.prevalidate_paths:
            validated_paths = []
            for i, path in enumerate(retained_paths):
                s = retained_path_idxes[i]
                e = retained_path_idxes[i+1]

                invalid_portions = np.where(path_state_validities[s:e] == False)[0]
                if len(invalid_portions) > 0:
                    validated_paths.append(Path(path.path[invalid_portions[-1]+1:]))
                else:
                    validated_paths.append(Path(path.path))
            self.validated_paths = validated_paths
        
        # new_db = Database()
        # new_db.paths = validated_paths
        # new_db.draw_paths(plt.gca())
        # self.env.draw_environment(plt.gca())
        # plt.scatter(start.value[0], start.value[1], color='green', s=100, zorder=2)
        # plt.scatter(target.value[0], target.value[1], color='red', s=100, zorder=2)
        # plt.show()
        # plt.clf()
        # self.validated_paths = validated_paths
        
        print(f"Number of Paths to Start with: {len(self.validated_paths)}")
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

    def get_tree_path_state_info(self):
        tree_states = np.array([key.value for key in self.tree])
        path_starting_idxes = np.array([len(path) for path in self.validated_paths])
        path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))
        path_states = np.array([state.value for path in self.validated_paths for state in path])
        return tree_states, path_starting_idxes, path_states

    def compute_c2g_estimates(self, tree_states, path_states):
        threshold = 5.0
        if len(path_states) == 0:
            # If no paths to follow, do rrt
            self.do_rrt = True
            return
        # print(tree_states.shape, path_states.shape)
        temp_mat = np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T))
        # print(np.min(temp_mat), np.max(temp_mat))
        # print(np.sort(temp_mat))
        tree_state_idx, path_state_idx = np.unravel_index(np.argmin(temp_mat), temp_mat.shape)

        # print(tree_states[tree_state_idx], path_states[path_state_idx])

        # dist_mat = np.sqrt(np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T)))
        dist_mat = np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T))

        dist_mat[dist_mat > (threshold**2)] = np.inf
        dist_mat[dist_mat == 0.0] = np.inf
        c2g_estimates = dist_mat + self.flattened_c2gs

        return c2g_estimates
    
    def filter_connection_attempts(self, c2g_estimates, tree_states, path_states):
        potential_connection_edges_idxes = np.where(c2g_estimates != np.inf)

        vals = c2g_estimates[potential_connection_edges_idxes[0], potential_connection_edges_idxes[1]]

        # num_connections_to_keep = 1500
        num_connections_to_keep = 4500
        # num_connections_to_keep = 9000

        best_n_connections = np.argsort(vals)[:num_connections_to_keep]
        other_n_connections = np.argsort(vals)[num_connections_to_keep:]

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
        return tree_state_idx, following_path, path_starting_idx

    def update_path_database(self, kept_path_segment, path_starting_idx):
        self.validated_paths[path_starting_idx] = Path([self.env.make_state(state) for state in kept_path_segment])

        if len(self.validated_paths[path_starting_idx]) == 0:
            self.validated_paths.pop(path_starting_idx)
    
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

    def filter_invalid_segments_of_follow_path(self, following_path, path_validity):
        invalid_idxes = np.where(path_validity == False)[0]
        if len(invalid_idxes) == 0:
            # IF HERE, SHOULD ALWAYS TAKE TREE STRAIGHT TO TARGET
            add_to_tree_segment = following_path
            kept_path_segment = following_path 
        else:
            invalid_idx = invalid_idxes[0]
            deletion_offset = 1
            add_to_tree_segment = following_path[:invalid_idx]
            kept_path_segment = following_path[invalid_idx+deletion_offset:]

        return add_to_tree_segment, kept_path_segment
    
    def add_states_from_path_to_tree(self, parent, add_to_tree_segment):

        for state in add_to_tree_segment:
            if np.all(parent.value == state): # HACK: This might be a hack (may need to figure out why this is happening)
                continue
            
            child = self.env.make_state(state)
            if child in self.tree: # HACK: This might be a hack (may need to figure out why this is happening)
                break

            self.tree[parent].append(child)
            self.tree[child]

            self.child_to_parent[child] = parent
            parent = child

    def step_search(self, iteration):
        tree_states, path_starting_idxes, path_states = self.get_tree_path_state_info()
        c2g_estimates = self.compute_c2g_estimates(tree_states, path_states)

        min_c2g_estimate = np.min(c2g_estimates)

        if min_c2g_estimate == np.inf or self.do_rrt:
            # Do RRT for a couple of steps
            rrt = RRT(self.env, delta=2)
            path_rrt = rrt.search(self.start, self.target, max_steps=10, starting_tree_info=(self.tree,self.child_to_parent))

            # path_rrt = rrt.search(start, target, max_steps=1000, starting_tree_info=(self.tree,self.child_to_parent))
            self.tree = rrt.tree
            self.child_to_parent = rrt.child_to_parent

            # print("RRTing")
            expansion_tech = 'rrt'
            self.do_rrt = False
        else:
            expansion_tech = 'pdg'

            c2g_estimates = self.filter_connection_attempts(c2g_estimates, tree_states, path_states)
            if np.min(c2g_estimates) == np.inf:
                self.do_rrt = True
                return

            tree_state_idx, following_path, path_starting_idx = self.get_follow_path(c2g_estimates, path_starting_idxes, path_states)

            path_validity = self.validate_follow_path(following_path)
            add_to_tree_segment, kept_path_segment = self.filter_invalid_segments_of_follow_path(following_path, path_validity)

            self.update_path_database(kept_path_segment, path_starting_idx)

            parent_node = self.env.make_state(tree_states[tree_state_idx])
            self.add_states_from_path_to_tree(parent_node, add_to_tree_segment)
        
        if self.target in self.tree:
            print(f"Found Path in {iteration} iterations")
            return
            
            # return self.backtrack(self.target)

        self.compute_c2g_for_paths(self.validated_paths, self.target)

    def init_search(self, start, target):
        self.start = start
        self.target = target

        self.tree = defaultdict(list)


        self.tree[self.start]
        self.child_to_parent = {}
        self.child_to_parent[self.start] = None
        self.do_rrt = False

    def search(self, start, target, max_steps=500):
        
        # self.start = start
        # self.target = target

        # self.tree = defaultdict(list)


        # self.tree[self.start]
        # self.child_to_parent = {}
        # self.child_to_parent[self.start] = None
        # self.do_rrt = False
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

class BiDirectionalPDG():
    def __init__(self, env, db_path):
        self.env = env
        self.db_path = db_path

        self.max_connection_distance = 0.5

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

    def backtrack(self, forward_pdg, backward_pdg, connection):
        if connection is None:
            return Path()
        forward_end_state, backward_end_state = connection
        forward_path = forward_pdg.backtrack(end=forward_end_state)
        backward_path = backward_pdg.backtrack(end=backward_end_state)

        # Join Both Paths and Reverse the backward PDG Tree path
        joined_path = forward_path.path + backward_path.path[::-1] 
        joined_path = self.delete_duplicate_states_in_path(joined_path)
        return Path(path=joined_path)

    def search(self, start, target, max_steps=500):
        self.start = start
        self.target = target


        self.forward_pdg = PDG(self.env, self.db_path)
        self.backward_pdg = PDG(self.env, self.db_path)

        

        self.forward_pdg.compute_retained_paths(self.target)
        self.backward_pdg.compute_retained_paths(self.start)

        self.forward_pdg.init_search(start, target)
        self.backward_pdg.init_search(target, start)

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

    # seed = 6172
    # seed = 641

    # Broken Seeds (on PC)
    # seed = 4459
    # seed = 9264 # Broken on Mac (Fixed?)
    # seed = 8718 # BUG SEED

    # seed = 1394 # BUG: Can't draw validated paths (Goal is not valid)

    # Testing from optimized_pdg.py
    # seed = 1136
    # seed = 1119
    # seed = 6473
    # seed = 277
    # seed = 7936
    # seed = 551

    ## (CHECK PATH LENGTH FOR BI)
    # seed = 1231 
    # seed = 6351 (Check why this is so slow for BiPDG)
    # seed = 896

    # Check Invalid values in SQRT
    # seed = 142
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
    # db_save_path = 'saves/database_v1_bpe3.pickle'
    # db_save_path = 'saves/clustered_database_large_bpe_subsampled.pickle'
    db_save_path = 'saves/database_rf2.pickle'
    
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    pdg = PDG(env, db_save_path)

    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([15.0, 5.0]))
    # start, target = env.make_state(np.array([5.0, 5.0])), env.make_state(np.array([45.0, 5.0]))
    start, target = env.space.sample_task()

    start_time = time.time()
    pdg.compute_retained_paths(target)
    end_time = time.time()
    print(f"Time to compute paths: {end_time-start_time}")

    start_time = time.time()
    path = pdg.search(start, target)
    end_time = time.time()
    print(f"Time to search: {end_time - start_time}", len(path))

    pdg.draw_tree(plt.gca(), path)
    plt.show()
    # exit()

    bipdg = BiDirectionalPDG(env, db_save_path)

    # start_time = time.time()
    # bipdg.compute_retained_paths(target, start)
    # end_time = time.time()
    # print(f"Time to compute paths (BiPDG): {end_time-start_time}")

    start_time = time.time()
    path = bipdg.search(start, target)
    end_time = time.time()
    print(f"Time to search (BiPDG): {end_time - start_time}", len(path))

    plt.clf()
    bipdg.forward_pdg.draw_tree(plt.gca(), path)
    bipdg.backward_pdg.draw_tree(plt.gca(), path)
    plt.show()

    # print([state.value for state in path])


    # for key in pdg.timing_dict:
    #     print(f"{key}: {np.sum(pdg.timing_dict[key])}")
    #     # print(pdg.timing_dict[key])

    

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