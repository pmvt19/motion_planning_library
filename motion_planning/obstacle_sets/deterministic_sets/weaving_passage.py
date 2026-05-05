from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d
from motion_planning.utils import create_rectangle_geometry


class WeavingPassage(ObstacleSet2d):
    def __init__(self):
        
        x_range = [0,10]
        y_range = [0,10]

        boundary = Polygon([[x_range[0], y_range[0]],
                            [x_range[0], y_range[1]],
                            [x_range[1], y_range[1]],
                            [x_range[1], y_range[0]]])
        
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