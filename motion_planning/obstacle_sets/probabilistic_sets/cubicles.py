import numpy as np
from shapely import Polygon

from motion_planning.obstacle_sets import ObstacleSet2d


class Cubicles(ObstacleSet2d):
    def __init__(self):

        self.obstacles = []
        self.central_points = []
        self.critical_points = []

        x_range = [0,45]
        y_range = [0,50]

        self.x_range = x_range
        self.y_range = y_range

        boundary = Polygon([(x_range[0], y_range[0]),
                            (x_range[0], y_range[1]),
                            (x_range[1], y_range[1]),
                            (x_range[1], y_range[0])])

        self.create_cubicle_sets()
        
        super().__init__(obstacles=self.obstacles, boundary=boundary)

    def create_single_cubicle(self, x_loc=0, y_loc=0, space_width=5, bottom_open=False):
        line_width = 0.5
        line_height = 6

        horizontal_obstacle = None
        
        if bottom_open:
            horizontal_obstacle = Polygon([ # Horizontal Bar
                    [x_loc+line_width, y_loc+line_height-line_width],
                    [x_loc+line_width, y_loc+line_height],
                    [x_loc+space_width+line_width, y_loc+line_height],
                    [x_loc+space_width+line_width, y_loc+line_height-line_width],
                ])
        else:
            horizontal_obstacle = Polygon([ # Horizontal Bar
                    [x_loc+line_width, y_loc+line_width],
                    [x_loc+line_width, y_loc],
                    [x_loc+space_width+line_width, y_loc],
                    [x_loc+space_width+line_width, y_loc+line_width],
                ])

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
            horizontal_obstacle
        ]
        x_center = (2 * x_loc + space_width + line_width*2) / 2
        y_center = (2 * y_loc + line_height - line_width) / 2

        sample_radius = 2
        space_samples = np.array([x_center, y_center]) + (
            np.random.normal(size=(1000, 2)) * sample_radius
        )

        self.central_points.append(np.array([x_center, y_center]))
        self.critical_points.extend(space_samples)

        return obs
    
    def create_single_cubicle_set(
        self, 
        x_loc, 
        y_loc, 
        space_width, 
        num_spaces, 
        bottom_open=False
    ):
        for i in range(num_spaces):
            self.obstacles.extend(
                self.create_single_cubicle((i * 5) + x_loc, 
                                           y_loc, 
                                           space_width, 
                                           bottom_open)
            )

    def get_x_loc_for_set(self, set_num, num_cubicles_per_set, cubicle_width):
        width_of_cubicle_set = ((cubicle_width + 1) * num_cubicles_per_set)
        return (width_of_cubicle_set * set_num) + 5 + (5 * set_num)

    def create_cubicle_sets(self, num_cubicles_per_set=4, cubicle_width=4):
        
        i = 0
        set_i_x_loc_vert = self.get_x_loc_for_set(i, num_cubicles_per_set, 
                                                  cubicle_width)

        while set_i_x_loc_vert < self.x_range[1]:
            for y_loc in [7, 21, 35]:
                self.create_single_cubicle_set(set_i_x_loc_vert, y_loc, cubicle_width, 
                                               num_cubicles_per_set, 
                                               np.random.randint(0, 2))

            i += 1
            set_i_x_loc_vert = self.get_x_loc_for_set(i, num_cubicles_per_set, 
                                                      cubicle_width)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obst_set = Cubicles()
    obst_set.draw(plt.gca())
    plt.show()