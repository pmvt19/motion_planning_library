from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d
from motion_planning.utils import create_rectangle_geometry


class CentralObstacle(ObstacleSet2d):
    def __init__(self, x_range=[0, 10], y_range=[0, 10]):
        obstacles = []
        obs = create_rectangle_geometry(5,5,2,2)
        obstacles.append(obs)

        boundary = Polygon([[x_range[0], y_range[0]],
                            [x_range[0], y_range[1]],
                            [x_range[1], y_range[1]],
                            [x_range[1], y_range[0]]])

        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = CentralObstacle()
    obst_set.draw(plt.gca())
    plt.show()