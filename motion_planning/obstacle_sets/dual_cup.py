import numpy as np

from motion_planning.obstacle_sets.obstacle_set_3d import ObstacleSet3D

class DualCup(ObstacleSet3D):
    def __init__(self):
        

        aa_rect_prism1 = np.array([0,0,0,1,5,5])
        aa_rect_prism2 = np.array([2.5,2.5,0,5,1,5])
        aa_rect_prism3 = np.array([2.5,-2.5,0,5,1,5])
        aa_rect_prism4 = np.array([2.5,0,-2.5,5,5,1])
        aa_rect_prism5 = np.array([2.5,0,2.5,5,5,1])

        prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3, aa_rect_prism4, aa_rect_prism5])
        self.obstacles = prisms

        super().__init__(obstacles=self.obstacles, boundary=None)


if __name__ == '__main__':
    obst_set = DualCup()
    obst_set.draw()