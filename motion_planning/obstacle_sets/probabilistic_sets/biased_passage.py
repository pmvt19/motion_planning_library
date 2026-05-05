import numpy as np
from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d


class BiasedPassage(ObstacleSet2d):
    def __init__(self, 
                 num_walls=1,
                 bias=0.5,
                 main_wall_width=2,
                 sup_wall_width=1,
                 gap_width=1):

        assert (num_walls > 0)

        obstacles = []
        boundary = []

        x_range = [0,(10 * (num_walls+1))]
        y_range = [0,10]

        boundary = Polygon([[0, 0],
                            [0, 10],
                            [x_range[1], 10],
                            [x_range[1], 0]])

        for i in range(num_walls):
            x_low = (10 * (i+1)) - main_wall_width/2
            x_high = (10 * (i+1)) + main_wall_width/2

            y_low = gap_width
            y_high = y_range[1] - gap_width
            
            obs = Polygon([[x_low, y_high],
                           [x_low, y_low],
                           [x_high, y_low],
                           [x_high, y_high]])
            obstacles.append(obs)

            sup_x_low = (10 * (i+1)) - sup_wall_width/2
            sup_x_high = (10 * (i+1)) + sup_wall_width/2

            if np.random.random() < bias:
                sup_y_low = y_range[0]
                sup_y_high = y_range[0] + gap_width
            else:
                sup_y_low = y_range[1] - gap_width
                sup_y_high = y_range[1]
            
            obs = Polygon([[sup_x_low, sup_y_low],
                           [sup_x_low, sup_y_high],
                           [sup_x_high, sup_y_high],
                           [sup_x_high, sup_y_low]])
            obstacles.append(obs)

        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = BiasedPassage(num_walls=3, bias=0.7)
    obst_set.draw(plt.gca())
    plt.show()