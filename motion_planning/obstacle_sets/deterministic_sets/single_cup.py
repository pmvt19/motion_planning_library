import numpy as np

from motion_planning.obstacle_sets.obstacle_set_3d import ObstacleSet3D


class SingleCup(ObstacleSet3D):
    def __init__(self):
        aa_rect_prism1 = np.array([2,0,1,0.2,1,1])
        aa_rect_prism2 = np.array([1.5,0.5,1,1,0.2,1])
        aa_rect_prism3 = np.array([1.5,-0.5,1,1,0.2,1])
        aa_rect_prism4 = np.array([1.5,0,0.5,1,1,0.2])
        aa_rect_prism5 = np.array([1.5,0,1.5,1,1,0.2])

        prisms = np.array(
            [
                aa_rect_prism1,
                aa_rect_prism2,
                aa_rect_prism3,
                aa_rect_prism4,
                aa_rect_prism5
            ]
        )
        prisms[:, 0] = prisms[:, 0] + 1 # Shift +1 units in the X Axis
        self.obstacles = prisms
        self.boundary = np.array([0,0,0,10,10,10])

        super().__init__(obstacles=self.obstacles, boundary=self.boundary)

if __name__ == '__main__':
    obst_set = SingleCup()
    obst_set.draw()