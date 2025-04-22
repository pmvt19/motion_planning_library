import numpy as np
import matplotlib.pyplot as plt
from rrt import RRT
from sklearn.neighbors import KDTree
import pickle

from space import PointRobot
from obstacle_sets import RandomSamplePassage

class BiasedSamplingRRT(RRT):
    def __init__(self, env, delta=0.5):
        super().__init__(env=env, delta=delta)
        self.sampling_path = pickle.load(open('saved_paths/rrt_path.pickle', 'rb'))

    def select_node(self, goal_bias=0):
        if np.random.random() < goal_bias:
            sampled_point = self.target
        else:
            # sampled_point = self.env.sample_valid_point()
            sampled_idx = np.random.choice(len(self.sampling_path))
            # print(self.sampling_path[sampled_idx].shape, np.random.normal(scale=0.5, size=(2,)).shape)
            sampled_point = self.sampling_path[sampled_idx].value + np.random.normal(scale=0.5, size=(2,))
            sampled_point = env.make_state(sampled_point)
        nodes = np.array([node.value for node in self.tree.keys()])
        kdt = KDTree(nodes)
        _, ind = kdt.query(np.array([sampled_point.value]), k=1)
        idx = ind[0][0]
        return self.env.make_state(nodes[idx]), sampled_point

if __name__ == '__main__':
    # np.random.seed(56)
    np.random.seed(8)

    env = PointRobot()
    env.set_obstacles(RandomSamplePassage())
    start, target = env.sample_task()

    rrt = BiasedSamplingRRT(env)
    path = rrt.search(start, target, max_steps=6000, goal_bias=0.1, animate_search_tree=False)



    env.draw_environment(plt.gca())
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()

