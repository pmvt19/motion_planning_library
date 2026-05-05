from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d


class TestSet(ObstacleSet2d):
    def __init__(self, x_range=[-10,10], y_range=[-10,10]):
        obstacles = [
            Polygon([
                [6, 6],
                [7, 6],
                [7, 7],
                [6, 7],            
            ]),
            Polygon([
                [2.5, -7.5],
                [5, -7.5],
                [5, 7.5],
                [2.5, 7.5],
            ])
        ]

        boundary = Polygon([(x_range[0], y_range[0]),
                            (x_range[0], y_range[1]),
                            (x_range[1], y_range[1]),
                            (x_range[1], y_range[0])])

        super().__init__(obstacles=obstacles, boundary=boundary)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = TestSet()
    obst_set.draw(plt.gca())
    plt.show()