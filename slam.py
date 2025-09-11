import numpy as np
import matplotlib.pyplot as plt
import copy

from space import RobotSpace, DiscRobot
from obstacle_sets import BiasedPassage

from lidar import Lidar
from occupancy_map import OccupancyMap


class SLAM():
    def __init__(self, env):
        self.env : RobotSpace = env
        self.om = OccupancyMap()

        os = BiasedPassage(num_walls=1)

        self.env.set_obstacles(os)

        self.lidar = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, os)

    def search_and_map(self, start, target):
        self.start = start
        self.target = target

        # for i in range(500):
        #     pass

        loc = self.start
        loc_state = self.start

        i = 0
        # while not np.isclose(loc.value, self.target):
        for _ in range(100):
            cur_path = self.om.search(loc_state, self.target)
            
            loc = self.om.idx_to_coord(np.array(cur_path[6]))
            # print(type(loc), loc)
            print(loc, np.array(cur_path[6]))
            loc_state = self.lidar.engine.make_state(loc)
            self.draw(loc_state, cur_path)
            
            plt.pause(0.5)

            readings = self.lidar.read_sensor(loc)
            self.om.update_map(self.lidar.engine.make_state(loc), readings)
            self.om.buffer_obstacles()

            self.draw(loc_state, cur_path)
            plt.pause(0.5)

            # i += 1
            # if i >= 2:
            #     break



    def draw(self, state=None, path=None):
        if path is None:
            fig, axs = plt.subplots(2)
            
            self.env.draw_environment(axs[0])
            if state:
                self.env.draw_state(axs[0], state)
            self.om.draw_map(axs[1])
        else:
            fig, axs = plt.subplots(3)

            self.env.draw_environment(axs[0])
            if state:
                self.env.draw_state(axs[0], state)
            self.om.draw_map(axs[1])

            temp_om = copy.deepcopy(self.om)
            temp_om.add_path_to_map(path)

            temp_om.draw_map(axs[2])



if __name__ == '__main__':
    env = DiscRobot()
    slam = SLAM(env=env) # internally takes care of Obstacle Set

    slam.draw()
    plt.show()

    start_np = np.array([5.0, 5.0])
    target_np = np.array([15.0, 5.0])

    start = env.make_state(start_np)
    target = env.make_state(target_np)

    slam.search_and_map(start, target)