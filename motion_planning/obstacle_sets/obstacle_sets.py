class ObstacleSet():
    def __init__(self, obstacles, boundary):
        self.obstacles = obstacles
        self.boundary = boundary
        self.central_points = []
        self.critical_points = []