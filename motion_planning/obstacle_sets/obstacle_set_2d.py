from motion_planning.obstacle_sets import ObstacleSet


class ObstacleSet2d(ObstacleSet):
    def __init__(self, obstacles, boundary):
        super().__init__(obstacles=obstacles, boundary=boundary)
    
    def draw(self, ax):
        x_points, y_points = self.boundary.exterior.xy
        x_range = [min(x_points), max(x_points)]
        y_range = [min(y_points), max(y_points)]
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        for obs in self.obstacles:
            x,y = obs.exterior.xy
            ax.plot(x,y, color='black')