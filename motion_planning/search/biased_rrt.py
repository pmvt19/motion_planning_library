import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.neighbors import KDTree

from motion_planning.rrt import RRT
from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import RandomSamplePassage, BiasedPassage

class BiasedSamplingRRT(RRT):
    def __init__(self, env, biased_points, points_bias=0.4, delta=0.5):
        super().__init__(env=env, delta=delta)
        # self.sampling_path = pickle.load(open('saved_paths/rrt_path.pickle', 'rb'))
        self.biased_points = biased_points
        self.points_bias = points_bias
        assert (len(self.biased_points) > 0), "Must have a greater than 0 number of biased points"

    # def select_node(self, goal_bias=0):
    #     if np.random.random() < goal_bias:
    #         sampled_point = self.target
    #     else:
    #         # sampled_point = self.env.sample_valid_point()
    #         sampled_idx = np.random.choice(len(self.sampling_path))
    #         # print(self.sampling_path[sampled_idx].shape, np.random.normal(scale=0.5, size=(2,)).shape)
    #         sampled_point = self.sampling_path[sampled_idx].value + np.random.normal(scale=0.5, size=(2,))
    #         sampled_point = self.env.make_state(sampled_point)
    #     nodes = np.array([node.value for node in self.tree.keys()])
    #     kdt = KDTree(nodes)
    #     _, ind = kdt.query(np.array([sampled_point.value]), k=1)
    #     idx = ind[0][0]
    #     return self.env.make_state(nodes[idx]), sampled_point

    def select_node(self, goal_bias=0):
        sampling_type = np.random.choice(a=['goal_biased', 'points_biased', 'uniform_random'], 
                                         p=[goal_bias, self.points_bias, (1-self.points_bias-goal_bias)])
        d = self.target.value.shape[0]

        if sampling_type == 'goal_biased':
            sampled_point = self.target
        elif sampling_type == 'points_biased':
            sampled_idx = np.random.choice(len(self.biased_points))
            sampled_point = self.biased_points[sampled_idx] + np.random.normal(scale=0.5, size=(d,)) # Unclear if the noise should be included
            sampled_point = self.env.make_state(sampled_point)
        elif sampling_type == 'uniform_random':
            sampled_point = self.env.sample_valid_point()

        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point

if __name__ == '__main__':
    # np.random.seed(56)
    np.random.seed(8)

    env = PointRobot()
    # env.set_obstacles(RandomSamplePassage())
    env.set_obstacles(BiasedPassage(num_walls=4))
    start, target = env.sample_task()

    rrt = BiasedSamplingRRT(env)
    path = rrt.search(start, target, max_steps=10, goal_bias=0.1, animate_search_tree=False)



    env.draw_environment(plt.gca())
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()

