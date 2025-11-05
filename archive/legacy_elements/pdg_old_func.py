def search_step(self, start, target):
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

        # print(following_path, path_state_validities)

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
        plt.clf()

        # print(potential_connection_edges_idxes)

        # print(self.flattened_c2gs)
        # flattened_c2gs = []

        # find closest on each path

        # find node with lowest c2g estimate and attach to the associated path (follow until collision)
    

### WORKING NON STEP FUNCTION SEARCH FUNCTION: 
def search(self, start, target):

        timing_dict = defaultdict(list)

        self.start = start
        self.target = target

        self.tree = defaultdict(list)


        self.tree[start]
        self.child_to_parent = {}
        self.child_to_parent[start] = None
        
        do_rrt = False
        for i in range(500):
            start_time = time.time()
            tree_states = np.array([key.value for key in self.tree])
            path_starting_idxes = np.array([len(path) for path in self.validated_paths])
            path_starting_idxes = np.cumsum((np.concatenate(([0], path_starting_idxes))))
            path_states = np.array([state.value for path in self.validated_paths for state in path])
            end_time = time.time()
            timing_dict['flattening_states_time'].append(end_time - start_time)

            # pairwise dist
            start_time = time.time()
            #### Compute tree node c2g estimates
            dist_mat = np.sqrt(np.sum(tree_states**2, axis=1, keepdims=True) + np.sum(path_states**2, axis=1, keepdims=True).T + (-2 * (tree_states @ path_states.T)))

            
            # threshold = 1.0
            threshold = 5.0
            # threshold = 0.5
            
            dist_mat[dist_mat > threshold] = np.inf
            dist_mat[dist_mat == 0.0] = np.inf
            
            c2g_estimates = dist_mat + self.flattened_c2gs
            #### Compute tree node c2g estimates
            end_time = time.time()
            timing_dict['dist_c2g_calc_time'].append(end_time - start_time)
            
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
            else:
                # print("PDGing")
                expansion_tech = 'pdg'

                #### Filter connection attempts
                potential_connection_edges_idxes = np.where(c2g_estimates != np.inf)

                vals = c2g_estimates[potential_connection_edges_idxes[0], potential_connection_edges_idxes[1]]

                # num_connections_to_keep = 1500
                num_connections_to_keep = 4500
                # num_connections_to_keep = 9000

                best_n_connections = np.argsort(vals)[:num_connections_to_keep]
                other_n_connections = np.argsort(vals)[num_connections_to_keep:]

                edge_starts = tree_states[potential_connection_edges_idxes[0][best_n_connections]]
                edge_ends = path_states[potential_connection_edges_idxes[1][best_n_connections]]

                start_time = time.time()
                edge_validities = self.env.batch_is_valid_edge_uniform(edge_starts, edge_ends)
                end_time = time.time()

                timing_dict['validate_potential_connection_time'].append(end_time - start_time)


                c2g_estimates[(potential_connection_edges_idxes[0][best_n_connections][edge_validities == False]), (potential_connection_edges_idxes[1][best_n_connections][edge_validities == False])] = np.inf
                c2g_estimates[(potential_connection_edges_idxes[0][other_n_connections]), (potential_connection_edges_idxes[1][other_n_connections])] = np.inf
                #### Filter connection attempts

                if np.min(c2g_estimates) == np.inf:
                    do_rrt = True
                    continue
                
                #### Pick Path to Follow
                tree_state_idx, path_state_idx = np.unravel_index(np.argmin(c2g_estimates), c2g_estimates.shape)

                path_starting_idx = np.where(path_state_idx < path_starting_idxes)[0][0] - 1

                following_path = path_states[path_state_idx:path_starting_idxes[path_starting_idx+1]]
                #### Pick Path to Follow
                start_time = time.time()

                #### Validate Path to Follow (Self-Contained)
                # # Inputs following_path
                path_validity = self.validate_follow_path(following_path)
                
                end_time = time.time()
                timing_dict['validate_follow_path'].append(end_time - start_time)
                # path_state_validities = self.env.batch_is_valid(following_path)
                

                # if len(following_path) == 1:
                #     path_edge_validities = []
                # else: 
                #     start_edge_states = following_path[:-1]
                #     end_edge_states = following_path[1:]

                #     path_edge_validities = self.env.batch_is_valid_edge(start_edge_states, end_edge_states)
                # end_time = time.time()
                # timing_dict['validate_follow_path'].append(end_time - start_time)
                

                # path_edge_validities = np.concatenate(([True], path_edge_validities))
                # path_validity = np.logical_and(path_state_validities, path_edge_validities)
                #### Validate Path to Follow

                #### Filter to Valid Segments of Path to Follow (Self-Contained)
                start_time = time.time()
                # Inputs needed (following_path, path_validity)
                # invalid_idxes = np.where(path_validity == False)[0]
                # if len(invalid_idxes) == 0:
                #     add_to_tree_segment = following_path
                #     kept_path_segment = following_path
                # else:
                #     invalid_idx = invalid_idxes[0]
                #     deletion_offset = 1
                #     add_to_tree_segment = following_path[:invalid_idx]
                #     kept_path_segment = following_path[invalid_idx+deletion_offset:] # TODO: Figure out logic for updating path states and all
                add_to_tree_segment, kept_path_segment = self.filter_invalid_segments_of_follow_path(following_path, path_validity)
                #### Filter to Valid Segments of Path to Follow

                end_time = time.time()
                timing_dict['filter_following_path'].append(end_time - start_time)

                #### Update Path to Follow in "Database" of Paths
                start_time = time.time()
                self.validated_paths[path_starting_idx] = Path([self.env.make_state(state) for state in kept_path_segment])
                
                if len(self.validated_paths[path_starting_idx]) == 0:
                    self.validated_paths.pop(path_starting_idx)
                end_time = time.time()
                timing_dict['update_saved_paths'].append(end_time - start_time)
                #### Update Path to Follow in "Database" of Paths

                #### Add Path to Follow States to Search Tree
                start_time = time.time()
                # parent = self.env.make_state(tree_states[tree_state_idx])
                # for state in add_to_tree_segment:
                #     if np.all(parent.value == state): # HACK: This might be a hack (may need to figure out why this is happening)
                #         continue
                    
                #     child = self.env.make_state(state)
                #     if child in self.tree:
                #         break

                #     self.tree[parent].append(child)
                #     self.tree[child]

                #     self.child_to_parent[child] = parent
                #     parent = child
                parent_node = self.env.make_state(tree_states[tree_state_idx])
                self.add_states_from_path_to_tree(parent_node, add_to_tree_segment)
                end_time = time.time()
                timing_dict['add_states_to_tree'].append(end_time - start_time)
                #### Add Path to Follow States to Search Tree
            
            if target in self.tree:
                print(f"Found Path in {i} iterations")
                self.timing_dict = timing_dict
                return self.backtrack(target)
            
            #### Recompute Cost-to-go (MODULARIZED)
            start_time = time.time()
            self.compute_c2g_for_paths(self.validated_paths, target)
            end_time = time.time()
            timing_dict['recompute_cost_to_gos'].append(end_time - start_time)
            #### Recompute Cost-to-go

            self.timing_dict = timing_dict