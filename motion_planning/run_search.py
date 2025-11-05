import numpy as np
from environments import CarParkingEnv, RandomSamplePassage, Environment2d
from prm import PRM, LazyPRM
from rrt import RRT 
from rsg import RandomSampleGeneration
import matplotlib.pyplot as plt
from utils import smooth_path, interpolate_edge
import pickle

if __name__ == '__main__':
    env = CarParkingEnv()
    # start, target = env.sample_task()
    start = env.make_state(np.array([2.0, 2.75, 0]))
    target = env.make_state(np.array([-3.0, -2.25, 0]))

    
    # rrt = RRT(env)
    # path = rrt.search(start, target, max_steps=10000, goal_bias=0.5)
    # rrt.draw_tree(plt.gca(), path=path, hold=True)
    
    # smoothed_path = smooth_path(env, path)
    # rrt.draw_tree(plt.gca(), path=smoothed_path, hold=True)

    # rsg = RandomSampleGeneration(env)
    # path = rsg.search(start, target, max_steps=10000, goal_bias=0.7)
    # rsg.draw_tree(plt.gca(), path=path, hold=True)
    # plt.show()
    
    # smoothed_path = smooth_path(env, path)
    # rsg.draw_tree(plt.gca(), path=smoothed_path, hold=True)


    # prm = PRM(env, num_samples=1000, num_neighbors=20, validate_edges=True)
    # prm.create_graph(starting_samples=env.parking_space_samples)
    # path = prm.search(start, target)
    # env.draw_environment(plt.gca())
    # prm.draw(plt.gca(), path=path)
    # plt.show()

    # env.draw_environment(plt.gca())
    # for p in path:
    #     env.draw_state(plt.gca(), p)
    # plt.show()
    # pickle.dump(path, open('saved_paths/path.pickle', 'wb'))
    # smoothed_path = smooth_path(env, path)
    # env.animate_path(path)
    # env.animate_path(smoothed_path)

    path = pickle.load(open('saved_paths/path.pickle', 'rb'))
    # path = [env.make_state(state) for int_edge in [interpolate_edge(path[i].value, path[i+1].value, 0.1) for i in range(len(path)-1)] for state in int_edge]
    # print(path)
    # exit()
    env.animate_path(path, frame_delay=0.1)
    # exit()

    from kinodynamic_optimization import traj_opt_smoothing
    smoothed_path = traj_opt_smoothing(env, path)
    env.animate_path(smoothed_path, frame_delay=0.5)

    print(smoothed_path)

    

