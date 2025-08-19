import numpy as np
import time
from collections import defaultdict
import pickle

from space import PointRobot
from circle_approximation import ApproximationSpace
from obstacle_sets import BiasedPassage

from rrt import RRT
from prm import PRM
from lightning import Lightning
from pdg import PDG
from optimized_pdg import OptimizedPDG

from database import Database

def run_experiment(seed, timing_dict, path_dict):
    np.random.seed(seed)
    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    start, target = env.sample_valid_point(), env.sample_valid_point()


    db_path = "saves/database_v1_bpe3.pickle"

    rrt = RRT(env)
    prm = PRM(env, num_samples=5000, num_neighbors=10, validate_edges=True)
    lightning = Lightning(env, db_path=db_path)
    pdg = PDG(env=env, db_path=db_path)
    optimized_pdg = OptimizedPDG(env=env, db_path=db_path)

    prm_create_graph_start = time.time()
    prm.create_graph()
    prm_create_graph_end = time.time()

    pdg_compute_retained_start = time.time()
    pdg.compute_retained_paths(target)
    pdg_compute_retained_end = time.time()

    optimized_pdg_compute_retained_start = time.time()
    optimized_pdg.compute_retained_paths(target)
    optimized_pdg_compute_retained_end = time.time()




    rrt_start_time = time.time()
    rrt_path = rrt.search(start, target)
    rrt_end_time = time.time()

    prm_start_time = time.time()
    prm_path = prm.search(start, target)
    prm_end_time = time.time()

    lightning_start_time = time.time()
    lightning_path = lightning.search(start, target)
    lightning_end_time = time.time()

    pdg_start_time = time.time()
    pdg_path = pdg.search(start, target)
    pdg_end_time = time.time()

    optimized_pdg_start_time = time.time()
    optimized_pdg_path = optimized_pdg.search(start, target)
    optimized_pdg_end_time = time.time()

    timing_dict['rrt_search_time'].append(rrt_end_time - rrt_start_time)
    timing_dict['prm_search_time'].append(prm_end_time - prm_start_time)
    timing_dict['lightning_search_time'].append(lightning_end_time - lightning_start_time)
    timing_dict['pdg_search_time'].append(pdg_end_time - pdg_start_time)
    timing_dict['optimized_pdg_search_time'].append(optimized_pdg_end_time - optimized_pdg_start_time)

    timing_dict['prm_create_graph'].append(prm_create_graph_end - prm_create_graph_start)

    timing_dict['pdg_retained_paths'].append(pdg_compute_retained_end - pdg_compute_retained_start)
    timing_dict['optimized_pdg_retained_paths'].append(optimized_pdg_compute_retained_end - optimized_pdg_compute_retained_start)

    path_dict['rrt_path'].append(rrt_path)
    path_dict['prm_path'].append(prm_path)
    path_dict['lightning_path'].append(lightning_path)
    path_dict['pdg_path'].append(pdg_path)
    path_dict['optimized_pdg_path'].append(optimized_pdg_path)    


def analyze_results(timing_dict, path_dict):
    for key in timing_dict:
        print(f"{key}:", np.mean(timing_dict[key]), np.median(timing_dict[key]), np.mean(np.sort(timing_dict[key])[:60]))

    for key in path_dict:
        lens = []
        for p in path_dict[key]:
            if p:
                lens.append(len(p))
            else:
                lens.append(0)
        print(f"{key}:", lens)



if __name__ == "__main__":
    timing_dict = defaultdict(list)
    path_dict = defaultdict(list)

    # for i in range(64):
    #     run_experiment(i, timing_dict, path_dict)


    timing_dict_path = "saves/timing_dict.pickle"
    path_dict_path = "saves/path_dict.pickle"

    # pickle.dump(timing_dict, open(timing_dict_path, 'wb'))
    # pickle.dump(path_dict, open(path_dict_path, 'wb'))

    timing_dict = pickle.load(open(timing_dict_path, 'rb'))
    path_dict = pickle.load(open(path_dict_path, 'rb'))

    analyze_results(timing_dict, path_dict)