from shapely import Polygon
from motion_planning.obstacle_sets import ObstacleSet2d
from motion_planning.utils import create_rectangle_geometry


class Shelves2d(ObstacleSet2d):
    def __init__(self):
        obstacles = []

        x_range = [-10,10]
        y_range = [-10,10]

        boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])

        super().__init__(obstacles=obstacles, boundary=boundary)

        self.obstacles.append(create_rectangle_geometry(x_loc=3.5, y_loc=2.5, x_width=4, y_length=1))
        self.obstacles.append(create_rectangle_geometry(x_loc=5.0, y_loc=0.5, x_width=1, y_length=3))
        self.obstacles.append(create_rectangle_geometry(x_loc=3.5, y_loc=-1.5, x_width=4, y_length=1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = Shelves2d()
    obst_set.draw(plt.gca())
    plt.show()