from motion_planning.search import PRM


class PRMStar(PRM):
    def __init__(
        self, env, num_samples=100, edge_dist_radius=None, cache_edge_validities=True
    ):
        super().__init__(
            env=env,
            num_samples=num_samples,
            num_neighbors=None,
            edge_dist_radius=edge_dist_radius,
        )
        self.cache_graph_edge_validities = cache_edge_validities
