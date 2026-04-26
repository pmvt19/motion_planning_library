import copy
import matplotlib.pyplot as plt

from motion_planning.space import RobotSpace, PolygonalRobot
from motion_planning.obstacle_sets import ObstacleSet, RandomSamplePassage

class MPSampler():
    def __init__(self, robot_space: RobotSpace, obstacle_set_constructor, obstacle_set_args):
        self.robot_space = robot_space

        # Function Pointer to the Constructor of the Obstacle Set
        self.obstacle_set_constructor = obstacle_set_constructor

        # Args for Obstacle Set as a Dictionary
        self.obstacle_set_args = obstacle_set_args

    def sample_env(self) -> RobotSpace:
        output_robot_space = copy.deepcopy(self.robot_space)
        output_robot_space.set_obstacles(self.obstacle_set_constructor(**self.obstacle_set_args))
        return output_robot_space

    def sample_task(self, env: RobotSpace):
        return env.sample_valid_point(), env.sample_valid_point()
    
if __name__ == '__main__':
    mp_sampler = MPSampler(PolygonalRobot(), RandomSamplePassage, {"num_walls": 3, "wall_width":1, "gap_width":1})

    for i in range(10):
        plt.cla()
        env = mp_sampler.sample_env()
        start, target = mp_sampler.sample_task(env)
        env.draw_environment(plt.gca())
        env.draw_state(plt.gca(), start)
        env.draw_state(plt.gca(), target)
        plt.show()
    
