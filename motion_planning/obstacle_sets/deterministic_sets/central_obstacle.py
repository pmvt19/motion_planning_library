from shapely import Polygon
from motion_planning.obstacle_sets import ObstacleSet2d
from motion_planning.utils import create_rectangle_geometry

# TODO: Fix range definitions
class CentralObstacle(ObstacleSet2d):
    def __init__(self):
        
        x_range = [0,10]
        y_range = [0,10]

        obstacles = []
        # obs = create_rectangle_geometry(5,5,5.1,5)
        obs = create_rectangle_geometry(5,5,2,2)
        # obs = create_rectangle_geometry(5,5,0.01,0.01)
        obstacles.append(obs)

        boundary = Polygon([[0, 0],
                            [0, 10],
                            [x_range[1], 10],
                            [x_range[1], 0]])

        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = CentralObstacle()
    obst_set.draw(plt.gca())
    plt.show()