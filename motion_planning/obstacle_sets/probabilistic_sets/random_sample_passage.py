import numpy as np
from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d


class RandomSamplePassage(ObstacleSet2d):
    def __init__(self, num_walls=3, wall_width=1, gap_width=1):
        obstacles = []
        boundary = []

        x_range = [0,(10 * (num_walls+1))]
        y_range = [0,10]

        boundary = Polygon([[0, 0],
                            [0, 10],
                            [x_range[1], 10],
                            [x_range[1], 0]])
        
        for i in range(num_walls):
            x_low = (10 * (i+1)) - wall_width/2
            x_high = (10 * (i+1)) + wall_width/2
            gap_y_loc = (
                np.random.random() * (y_range[1] - y_range[0] - gap_width)
                + y_range[0] + gap_width/2
            )

            y_low = gap_y_loc - gap_width/2
            y_high = gap_y_loc + gap_width/2
            
            obs = Polygon([[x_low, y_range[0]],
                           [x_low, y_low],
                           [x_high, y_low],
                           [x_high, y_range[0]]])
            obstacles.append(obs)
            
            obs = Polygon([[x_low, y_range[1]],
                           [x_low, y_high],
                           [x_high, y_high],
                           [x_high, y_range[1]]])
            obstacles.append(obs)

        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = RandomSamplePassage(num_walls=2, wall_width=2, gap_width=1)
    obst_set.draw(plt.gca())
    plt.show()