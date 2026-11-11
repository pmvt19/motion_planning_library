import rerun as rr

from motion_planning.obstacle_sets import ObstacleSet


class ObstacleSet3D(ObstacleSet):
    def __init__(self, obstacles, boundary):
        super().__init__(obstacles=obstacles, boundary=boundary)
    
    def draw(self):
        rr.init("Obstacle Set", spawn=True)

        for i, obst in enumerate(self.obstacles):
            centers = obst[:3]
            half_sizes = obst[3:] / 2
            rr.log(
                f"obsts_{i}", 
                rr.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    colors=[0, 0, 255],
                    fill_mode=rr.components.FillMode.Solid
                )
            )

if __name__ == "__main__":
    obst = ObstacleSet3D(None, None)
    obst.draw()