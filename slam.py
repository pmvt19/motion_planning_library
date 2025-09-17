import numpy as np
import matplotlib.pyplot as plt
import copy

from space import RobotSpace, DiscRobot
from obstacle_sets import BiasedPassage, RandomSamplePassage

from lidar import Lidar, OptimizedLidar, SuperOptimizedLidar
from occupancy_map import OccupancyMap


class SLAM():
    def __init__(self, env : RobotSpace):
        self.env : RobotSpace = env
        self.om = OccupancyMap()

        os = BiasedPassage(num_walls=2)
        # os = RandomSamplePassage(num_walls=2)

        self.env.set_obstacles(os)

        # self.lidar = Lidar((0.01, 0.1), (0, 2*np.pi), 100, 10.0, os)
        # self.lidar = OptimizedLidar((0.01, 0.1), (0, 2*np.pi), 100, 4.9, os)
        self.lidar = SuperOptimizedLidar(None, (0, 2*np.pi), 100, 4.9, os)
        # self.lidar = OptimizedLidar((0.01, 0.1), (0, 2*np.pi), 360, 4.9, os)

    def search_and_map(self, start, target):
        self.start = start
        self.target = target

        # for i in range(500):
        #     pass

        loc = self.start
        loc_state = self.start

        i = 0
        # while not np.isclose(loc.value, self.target):

        cur_path = self.om.search(loc_state, self.target)

        visited = set()
        # TODO: Temporary Solution to Plotting Issue
        fig, axs = plt.subplots(3)
        print(len(axs))
        for _ in range(150):
            

            # if cur_path:
            # choose_idx = 10 if len(cur_path) >= 11 else -1
            # choose_idx = 10 if len(cur_path) >= 11 else -1
            choose_idx = 4 if len(cur_path) >= 5 else -1
            # choose_idx = 1

            # while cur_path[choose_idx] in visited:
            #     choose_idx += 1
            # visited.add(cur_path[choose_idx])

            loc = self.om.idx_to_coord(np.array(cur_path[choose_idx]))

            # else:
            #     self.draw(loc_state)
            #     plt.show()
            #     exit()
            loc_state = self.lidar.engine.make_state(loc)

            # self.draw(loc_state, cur_path)
            # plt.pause(0.5)

            readings = self.lidar.read_sensor(loc)
            self.om.update_map(self.lidar.engine.make_state(loc), readings)
            self.om.buffer_obstacles(spread_value=0.9)
            cur_path = self.om.search(loc_state, self.target)
            # plt.clf()
            self.draw(axs, loc_state, cur_path)
            plt.pause(0.5)

            axs[0].clear()
            axs[1].clear()
            axs[2].clear()
            # plt.show()
            # plt.clf()
            # plt.cla()

            # i += 1
            # if i >= 2:
            #     break



    def draw(self, axs, state=None, path=None):
        if path is None:
            # fig, axs = plt.subplots(2)
            
            self.env.draw_environment(axs[0])
            if state:
                self.env.draw_state(axs[0], state)
            self.om.draw_map(axs[1])
        else:
            # fig, axs = plt.subplots(3)

            self.env.draw_environment(axs[0])
            if state:
                self.env.draw_state(axs[0], state)
            self.om.draw_map(axs[1])

            temp_om = copy.deepcopy(self.om)
            temp_om.add_path_to_map(path)

            temp_om.draw_map(axs[2])



if __name__ == '__main__':

    # np.random.seed(1)

    # np.random.seed(1) # Working Seed on BPE wall = 1 (Standard Far Task)
    # np.random.seed(2) # Working seed on BPE Wall = 2 (Standard Far Task)

    env = DiscRobot(disc_radius=0.2)
    slam = SLAM(env=env) # internally takes care of Obstacle Set

    # slam.draw()
    # plt.show()
    # start_np = np.array([14.72, 9.72])
    # start_np = np.array([14.72, 9.72])
    start_np = np.array([5.0, 5.0])
    target_np = np.array([25.0, 5.0])

    start_idx = slam.om.coord_to_idx(start_np)
    start_coord = slam.om.idx_to_coord(start_idx)

    print(start_np, start_idx, start_coord)

    start = env.make_state(start_np)
    target = env.make_state(target_np)

    slam.search_and_map(start, target)