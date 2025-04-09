from shapely import Polygon
import numpy as np


class ObstacleSet():
    def __init__(self, obstacles, boundary):
        self.obstacles = obstacles
        self.boundary = boundary
        self.critical_points = []

class ObstacleSet2d(ObstacleSet):
    def __init__(self, obstacles, boundary):
        super().__init__(obstacles=obstacles, boundary=boundary)
    
class TestSet(ObstacleSet2d):
    def __init__(self):
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

        x_range = [-10,10]
        y_range = [-10,10]

        boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])

        super().__init__(obstacles=obstacles, boundary=boundary)

class ParkingSpace(ObstacleSet2d):
    def __init__(self):
        self.obstacles = []
        
        self.parking_space_samples = []
        self.parking_space_centers = []

        x_range = [-15,15]
        y_range = [-15,15]
        self.boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])
        self.angular_dims_start = 2

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

        self.parking_space_centers.append(np.array([x_center, y_center, 0.0]))

        sample_radius = 2
        space_samples = np.array([x_center, y_center, 0]) + (np.random.normal(size=(1000, 3)) * sample_radius)
        self.parking_space_samples.extend(space_samples)

        return obs
        
if __name__ == '__main__':
    obs_set = TestSet()