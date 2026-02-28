import numpy as np
import matplotlib.pyplot as plt

from motion_planning.space import PointRobot, DiscRobot, PolygonalRobot, PlanarMobileArm
from motion_planning.circle_approximation import ApproximationSpace
from motion_planning.obstacle_sets import BiasedPassage, RandomSamplePassage
from motion_planning.prm import PRM
from motion_planning.utils import set_numpy_seed, smooth_path, interpolate_path

def save_animated_path_frames(save_dir, env, path):
    for i, state in enumerate(path):
        plt.clf()
        env.draw_environment(plt.gca())
        env.draw_state(plt.gca(), state)
        plt.savefig(f"{save_dir}/step_{i}.png")

def generate_point_robot_figs(save_figs=False):
    set_numpy_seed()

    os = BiasedPassage()
    env = PointRobot()
    env.set_obstacles(os)
    env = ApproximationSpace(env, do_overapproximation=True)

    start, target = env.make_state(np.array([2.5, 5.0])), env.make_state(np.array([17.5, 5.0]))
    prm = PRM(env, num_samples=1000, num_neighbors=10, validate_edges=True)
    prm.create_graph()

    path = prm.search(start, target)
    smoothed_and_interpolated_path = interpolate_path(smooth_path(env, path), env, delta=0.1)
    print(f"Final Path Length: {len(smoothed_and_interpolated_path)}")
    if save_figs:
        save_animated_path_frames('saves/robots/point_robot/', env.space, smoothed_and_interpolated_path)

def generate_disc_robot_figs(save_figs=False):
    set_numpy_seed(2243)

    os = RandomSamplePassage(num_walls=1, gap_width=4)
    env = DiscRobot(disc_radius=1.75)
    env.set_obstacles(os)
    env = ApproximationSpace(env, do_overapproximation=True)

    start, target = env.make_state(np.array([2.5, 5.0])), env.make_state(np.array([17.5, 5.0]))
    prm = PRM(env, num_samples=40000, num_neighbors=10, validate_edges=True)
    prm.create_graph()

    path = prm.search(start, target)
    smoothed_and_interpolated_path = interpolate_path(smooth_path(env, path), env, delta=0.1)
    print(f"Final Path Length: {len(smoothed_and_interpolated_path)}")

    if save_figs:
        save_animated_path_frames('saves/robots/disc_robot/', env.space, smoothed_and_interpolated_path)
    else:
        env.space.animate_path(smoothed_and_interpolated_path)

def generate_polygonal_robot_figs():
    set_numpy_seed(4909)

    os = RandomSamplePassage(num_walls=2, gap_width=1.2)
    env = PolygonalRobot()
    env.set_obstacles(os)
    env = ApproximationSpace(env, do_overapproximation=True)

    start, target = env.make_state(np.array([2.5, 5.0, 0.0])), env.make_state(np.array([27.5, 5.0, 0.0]))
    prm = PRM(env, num_samples=10000, num_neighbors=10, validate_edges=True)
    prm.create_graph()

    path = prm.search(start, target)
    smoothed_and_interpolated_path = interpolate_path(smooth_path(env, path), env, delta=0.1)
    print(f"Final Path Length: {len(smoothed_and_interpolated_path)}")
    save_animated_path_frames('saves/robots/polygonal_robot/', env.space, smoothed_and_interpolated_path)

def generate_planer_mobile_arm_figs(save_figs=False):
    set_numpy_seed(980)

    os = RandomSamplePassage(num_walls=1, gap_width=2.5)
    env = PlanarMobileArm()
    env.set_obstacles(os)
    env = ApproximationSpace(env, do_overapproximation=True)

    start = env.sample_valid_point()
    target = env.sample_valid_point()

    start = env.make_state(np.array([2.5, 5.0, np.pi/2, np.pi+(np.pi/4), np.pi+(np.pi/4)]))
    target = env.make_state(np.array([17.5, 5.0, np.pi/2, np.pi-(np.pi/4), np.pi-(np.pi/4)]))

    prm = PRM(env, num_samples=4000, num_neighbors=10, validate_edges=True)
    prm.create_graph()

    path = prm.search(start, target)
    smoothed_and_interpolated_path = interpolate_path(smooth_path(env, path), env, delta=0.1)
    print(f"Final Path Length: {len(smoothed_and_interpolated_path)}")

    if save_figs:
        save_animated_path_frames('saves/robots/planar_mobile_arm_robot/', env.space, smoothed_and_interpolated_path)
    else:
        env.space.animate_path(smoothed_and_interpolated_path)
        # env.animate_path(smoothed_and_interpolated_path)

if __name__ == '__main__':
    # generate_polygonal_robot_figs()
    # generate_disc_robot_figs(save_figs=True)
    # generate_planer_mobile_arm_figs(save_figs=True)
    generate_point_robot_figs(save_figs=True)
    


    
    