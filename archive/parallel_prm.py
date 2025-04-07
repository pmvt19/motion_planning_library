from prm import PRM
import threading

class ParallelPRM(PRM):
    def __init__(self, env, num_samples=10, num_neighbors=10, validate_edges=False):
        super().__init__(env=env, num_samples=num_samples, num_neighbors=num_neighbors, validate_edges=validate_edges)

    def validate_graph_edges_parallel(self, start_idx, end_idx):
        for a in range(start_idx, end_idx):
            for i, b in enumerate(self.graph.edges[a]):
                if not self.env.is_valid_edge(self.graph.vertices[a], self.graph.vertices[b]):
                    self.graph.edges[a, i] = -1    


    def validate_graph_edges(self):
        threads = []
        splits = list(range(0, len(self.graph.edges), 20)) + [len(self.graph.edges)]
        print(splits)
        for i in range(len(splits)-1):
            t = threading.Thread(target=self.validate_graph_edges_parallel, args=[splits[i], splits[i+1]])
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        # half_edges = int(len(self.graph.edges) // 2)
        # t1 = threading.Thread(target=self.validate_graph_edges_parallel, args=[0, half_edges])
        # t2 = threading.Thread(target=self.validate_graph_edges_parallel, args=[half_edges, len(self.graph.edges)])
        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()



        # # self.invalid_edges = []
        # for a, neighbors in enumerate(self.graph.edges):
        #     for i, b in enumerate(neighbors):
        #         if not self.env.is_valid_edge(self.graph.vertices[a], self.graph.vertices[b]):
        #             self.graph.edges[a, i] = -1
        #             # self.invalid_edges.append((self.graph.vertices[a], self.graph.vertices[b]))