from shapely import Polygon

class ObstacleSet():
    def __init__(self, obstacles, boundary):
        self.obstacles = obstacles
        self.boundary = boundary

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
        
if __name__ == '__main__':
    obs_set = TestSet()