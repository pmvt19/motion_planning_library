import numpy as np
from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d
from motion_planning.utils import create_rectangle_geometry


class BugTrap(ObstacleSet2d):
    def __init__(self, x_range=[-10, 10], y_range=[-10, 10], gap_width: float = 1.5):
        obstacles = []

        boundary = Polygon(
            [
                (x_range[0], y_range[0]),
                (x_range[0], y_range[1]),
                (x_range[1], y_range[1]),
                (x_range[1], y_range[0]),
            ]
        )

        super().__init__(obstacles=obstacles, boundary=boundary)
        # TODO: Make Gap Width Configurable
        # TODO: Migrate to anchored geometry configuration
        # TODO: Add ability to rotate and offset the bugtrap

        # Top & Bottom Walls
        self.obstacles.append(
            create_rectangle_geometry(x_loc=0.0, y_loc=4, x_width=6, y_length=1)
        )
        self.obstacles.append(
            create_rectangle_geometry(x_loc=0.0, y_loc=-4, x_width=6, y_length=1)
        )

        # Rear Wall
        self.obstacles.append(
            create_rectangle_geometry(x_loc=3.5, y_loc=0, x_width=1.0, y_length=9)
        )

        # Front Trap Walls
        self.obstacles.append(
            create_rectangle_geometry(
                x_loc=-3.25, y_loc=2.75, x_width=0.5, y_length=3.5
            )
        )
        self.obstacles.append(
            create_rectangle_geometry(
                x_loc=-3.25, y_loc=-2.75, x_width=0.5, y_length=3.5
            )
        )

        # Inner Trap Walls
        self.obstacles.append(
            create_rectangle_geometry(
                x_loc=-2.25, y_loc=-0.75, x_width=2.5, y_length=0.5
            )
        )
        self.obstacles.append(
            create_rectangle_geometry(
                x_loc=-2.25, y_loc=0.75, x_width=2.5, y_length=0.5
            )
        )

        self.critical_points.append(np.array([0.0, 0.0]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    obst_set = BugTrap()
    obst_set.draw(plt.gca())
    plt.show()
