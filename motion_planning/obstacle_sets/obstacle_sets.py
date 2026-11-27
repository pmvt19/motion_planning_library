class ObstacleSet():
    def __init__(self, obstacles: list, boundary):
        self.obstacles: list = obstacles
        self.boundary = boundary
        self.central_points: list = []
        self.critical_points: list = []