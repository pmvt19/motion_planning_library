import numpy as np

from shapely import Polygon
from motion_planning.obstacle_sets import ObstacleSet2d
from motion_planning.utils import create_rectangle_geometry

# TODO: Fix this Implementation
# Why is y_range not used?
# Num blocks should be a parameter
class WeavingPassage(ObstacleSet2d):
    def __init__(self):
        
        x_range = [0,10]
        y_range = [0,10]

        boundary = Polygon([[0, 0],
                            [0, 10],
                            [x_range[1], 10],
                            [x_range[1], 0]])
        
        obstacles = []

        num_blocks = 9

        for i in range(num_blocks):
            x_loc = 3 if i % 2 == 0 else 7
            obstacles.append(create_rectangle_geometry(x_loc, i+1, 7, 0.9))

        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = WeavingPassage()
    obst_set.draw(plt.gca())
    plt.show()