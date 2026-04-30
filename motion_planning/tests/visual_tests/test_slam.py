import numpy as np

from motion_planning.space import DiscRobot
from motion_planning.mapping import SLAM

def run_slam():
    env = DiscRobot(disc_radius=0.2)
    slam = SLAM(env=env) # internally takes care of Obstacle Set

    # Define Task
    start_np = np.array([5.0, 5.0])
    target_np = np.array([25.0, 5.0])

    start = env.make_state(start_np)
    target = env.make_state(target_np)

    slam.search_and_map(start, target)

if __name__ == '__main__':
    run_slam()