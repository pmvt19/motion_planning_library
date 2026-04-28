import numpy as np

from shapely import Polygon
from motion_planning.obstacle_sets import ObstacleSet2d

class ParkingSpace(ObstacleSet2d):
    def __init__(self):
        obstacles = []
        x_range = [-15,15]
        y_range = [-15,15]
        boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])
        super().__init__(obstacles=obstacles, boundary=boundary)

        self.obstacles.extend(self.create_parking_space(space_width=5))
        self.obstacles.extend(self.create_parking_space(x_loc=-7.5, y_loc=-7.5, space_width=5))
    
    def create_parking_space(self, x_loc=0, y_loc=0, space_width=5):
        line_width = 0.5
        line_height = 6
        obs = [
            Polygon([
                [x_loc, y_loc],
                [x_loc, y_loc+line_height],
                [x_loc+line_width, y_loc+line_height],
                [x_loc+line_width, y_loc],            
            ]),
            Polygon([
                [x_loc+space_width+line_width, y_loc],
                [x_loc+space_width+line_width, y_loc+line_height],
                [x_loc+space_width+line_width*2, y_loc+line_height],
                [x_loc+space_width+line_width*2, y_loc],            
            ]),
            Polygon([ # Horizontal Bar
                [x_loc+line_width, y_loc+line_height-line_width],
                [x_loc+line_width, y_loc+line_height],
                [x_loc+space_width+line_width, y_loc+line_height],
                [x_loc+space_width+line_width, y_loc+line_height-line_width],            
            ]),
        ]
        x_center = (2 * x_loc + space_width + line_width*2) / 2
        y_center = (2 * y_loc + line_height - line_width) / 2

        sample_radius = 2
        space_samples = np.array([x_center, y_center]) + (np.random.normal(size=(1000, 2)) * sample_radius)

        self.central_points.append(np.array([x_center, y_center]))
        self.critical_points.extend(space_samples)

        return obs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = ParkingSpace()
    obst_set.draw(plt.gca())
    plt.show()