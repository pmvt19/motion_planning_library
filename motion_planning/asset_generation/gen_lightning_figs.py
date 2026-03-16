import numpy as np
import matplotlib.pyplot as plt

from motion_planning.database import Database
from motion_planning.lightning import Lightning
from motion_planning.space import PointRobot
from motion_planning.circle_approximation import ApproximationSpace
from motion_planning.obstacle_sets import BiasedPassage
from motion_planning.rrt import RRT
from motion_planning.utils import smooth_path
from motion_planning.path import Path

def compute_candidate_paths(lightning, start, target, n=5):
    path_starts = np.array([path[0].value for path in lightning.db.paths])
    path_targets = np.array([path[-1].value for path in lightning.db.paths])

    start_dists = np.linalg.norm(path_starts - start.value, axis=1)
    target_dists = np.linalg.norm(path_targets - target.value, axis=1)

    summed_dists = start_dists + target_dists
    ordered_path_idxes = np.argsort(summed_dists)

    best_n_path_idxes = ordered_path_idxes[:n]

    path_lengths = [0] + [len(path) for path in lightning.db.paths]
    path_idxes = np.cumsum(path_lengths)

    all_path_states = np.array([state.value for path in lightning.db.paths for state in path])
    return best_n_path_idxes, path_idxes, all_path_states

def draw_candidate_paths(save_fig=False):
    plt.cla()
    np.random.seed(0)

    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    db_path = "saves/database_bpe3_large.pickle"
    lightning = Lightning(env=env, db_path=db_path)

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([35.0, 7.0]))

    best_n_path_idxes, path_idxes, all_path_states = compute_candidate_paths(lightning, start, target)
    env.space.draw_environment(plt.gca())
    plt.scatter(*start.value, color='green', s=100, zorder=2)
    plt.scatter(*target.value, color='red', s=100, zorder=2)
    for path_idx in best_n_path_idxes:
        start_idx = path_idxes[path_idx]
        end_idx = path_idxes[path_idx+1]

        path = all_path_states[start_idx:end_idx]
        plt.plot(path[:, 0], path[:, 1], marker='o')
    
    if save_fig:
        plt.savefig("saves/lightning/candidate_paths.png") 
    else:
        plt.show()

def draw_selected_path(save_fig=False):
    plt.cla()
    np.random.seed(0)

    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    db_path = "saves/database_bpe3_large.pickle"
    lightning = Lightning(env=env, db_path=db_path)

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([35.0, 7.0]))

    path_idx, path_validity = lightning.compute_candidate_paths(start, target)
    env.space.draw_environment(plt.gca())
    plt.scatter(*start.value, color='green', s=100, zorder=2)
    plt.scatter(*target.value, color='red', s=100, zorder=2)

    path = lightning.db.paths[path_idx]
    numpy_path = np.array([state.value for state in path])
    plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o')

    invalid_path_states_mask = path_validity == False
    invalid_path_states = numpy_path[invalid_path_states_mask]
    plt.plot(invalid_path_states[:, 0], invalid_path_states[:, 1], marker='o', color="#C00A0A", zorder=2, label='Invalid Segments')
    # plt.legend()
    
    if save_fig:
        plt.savefig("saves/lightning/selected_path.png") 
    else:
        plt.show()

def repair_path(lightning, path, path_validities):
    path = path.path

    num_invalid = 0
    prev = True
    intervals_start = []
    intervals_end = []
    
    for idx, validity in enumerate(path_validities):
        if prev != validity and validity == False:
            num_invalid += 1
            intervals_start.append(idx)
        elif prev == False and validity == True:
            intervals_end.append(idx)
        prev = validity
    
    intervals = list(zip(intervals_start, intervals_end))

    repaired_segments = []
    rrt = RRT(lightning.env)

    for seg_start_idx, seg_end_idx in intervals:

        repaired_segment = rrt.search(path[seg_start_idx-1], path[seg_end_idx], max_steps=lightning.max_repair_steps)
        repaired_segments.append(repaired_segment)

    for i in reversed(range(len(intervals))):
        seg_start_idx, seg_end_idx = intervals[i]
        path = path[0:seg_start_idx] + repaired_segments[i].path + path[seg_end_idx+1:]

    start_to_path = rrt.search(lightning.start, path[0], max_steps=lightning.max_repair_steps)
    path_to_target = rrt.search(path[-1], lightning.target, max_steps=lightning.max_repair_steps)
    
    return start_to_path.path + path + path_to_target.path, repaired_segments, start_to_path, path_to_target

def draw_path_repair(save_fig=False):
    plt.cla()
    np.random.seed(0)

    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    db_path = "saves/database_bpe3_large.pickle"
    lightning = Lightning(env=env, db_path=db_path)

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([35.0, 7.0]))
    lightning.start = start 
    lightning.target = target

    path_idx, path_validity = lightning.compute_candidate_paths(start, target)
    path = lightning.db.paths[path_idx]

    final_path, repaired_segments, start_to_path, path_to_target = repair_path(lightning, path, path_validity)
    env.space.draw_environment(plt.gca())
    plt.scatter(*start.value, color='green', s=100, zorder=2)
    plt.scatter(*target.value, color='red', s=100, zorder=2)

    # Plot Repaired Segments
    for repaired_segment in repaired_segments:
        numpy_path = np.array([state.value for state in repaired_segment])
        plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o', label='Repaired Segments')

    # Plot Connection to Task Segments
    numpy_path = np.array([state.value for state in start_to_path])
    plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o', label='Start to Path Segment')

    numpy_path = np.array([state.value for state in path_to_target])
    plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o', label='Path to Target Segment')

    # Plot Final Path
    numpy_path = np.array([state.value for state in final_path])
    plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o', zorder=0, label='Path From Database')

    plt.legend()
    
    if save_fig:
        plt.savefig("saves/lightning/repaired_path.png")
    else:
        plt.show()

def draw_final_path(save_fig=False):
    plt.cla()
    np.random.seed(0)

    env = PointRobot()
    env.set_obstacles(BiasedPassage(bias=0.5, num_walls=3))
    env = ApproximationSpace(env, batch_size=1000, do_overapproximation=True)

    db_path = "saves/database_bpe3_large.pickle"
    lightning = Lightning(env=env, db_path=db_path)

    start = env.make_state(np.array([5.0, 5.0]))
    target = env.make_state(np.array([35.0, 7.0]))
    lightning.start = start 
    lightning.target = target

    path_idx, path_validity = lightning.compute_candidate_paths(start, target)
    path = lightning.db.paths[path_idx]

    final_path, _, _, _ = repair_path(lightning, path, path_validity)

    env.space.draw_environment(plt.gca())
    plt.scatter(*start.value, color='green', s=100, zorder=2)
    plt.scatter(*target.value, color='red', s=100, zorder=2)

    # Plot Final Path
    numpy_path = np.array([state.value for state in final_path])
    plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o', zorder=0, label='Final Path')
    
    if save_fig:
        plt.savefig("saves/lightning/final_path.png")
    else:
        plt.show()

    plt.cla()
    env.space.draw_environment(plt.gca())
    smoothed_path = smooth_path(env, Path(final_path))
    numpy_path = np.array([state.value for state in smoothed_path])
    plt.scatter(*start.value, color='green', s=100, zorder=2)
    plt.scatter(*target.value, color='red', s=100, zorder=2)
    plt.plot(numpy_path[:, 0], numpy_path[:, 1], marker='o', zorder=0, label='Final Path')

    if save_fig:
        plt.savefig("saves/lightning/smoothed_final_path.png")
    else:
        plt.show()
    


# draw_candidate_paths(save_fig=True)
# draw_selected_path(save_fig=True)
# draw_path_repair(save_fig=True)
draw_final_path(save_fig=True)
