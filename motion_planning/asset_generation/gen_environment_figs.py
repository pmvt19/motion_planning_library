import matplotlib.pyplot as plt

from motion_planning.space import PointRobot
from motion_planning.obstacle_sets import ParkingSpace, BiasedPassage, RandomSamplePassage


def generate_robot_environment(environment_name=None):
    obstacle_set = None
    if environment_name == 'ParkingSpace':
        obstacle_set = ParkingSpace()
    elif environment_name == 'BiasedPassage':
        obstacle_set = BiasedPassage()
    elif environment_name == 'RandomSamplePassage':
        obstacle_set = RandomSamplePassage()
    elif environment_name is None:
        raise NotImplementedError

    env = PointRobot()
    env.set_obstacles(obstacle_set)
    return env
if __name__ == '__main__':
    ps_env = generate_robot_environment(environment_name='ParkingSpace')
    bp_env = generate_robot_environment(environment_name='BiasedPassage')
    rsp_env = generate_robot_environment(environment_name='RandomSamplePassage')

    plt.clf()
    ps_env.draw_environment(plt.gca())
    plt.savefig('saves/environments/parking_space.png')

    plt.clf()
    bp_env.draw_environment(plt.gca())
    plt.savefig('saves/environments/biased_passage.png')

    plt.clf()
    rsp_env.draw_environment(plt.gca())
    plt.savefig('saves/environments/random_sample_passage.png')
