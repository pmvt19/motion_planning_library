
from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d


class NonRegularPolygonObst(ObstacleSet2d):
    def __init__(self):
        obstacles = []

        obs = Polygon([
            (0,0),
            (2,1),
            (1.6,3),
            (1.3,3.5),
            (1.2,3),
            (-0.1,1.3)
        ])

        obstacles.append(obs)

        x_range = [-10,10]
        y_range = [-10,10]

        boundary = Polygon([[x_range[0], y_range[0]],
                            [x_range[0], y_range[1]],
                            [x_range[1], y_range[1]],
                            [x_range[1], y_range[0]]])
        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = NonRegularPolygonObst()
    obst_set.draw(plt.gca())
    plt.show()