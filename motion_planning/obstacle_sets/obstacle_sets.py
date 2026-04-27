import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon

from motion_planning.utils import create_rectangle_geometry


class ObstacleSet():
    def __init__(self, obstacles, boundary):
        self.obstacles = obstacles
        self.boundary = boundary
        self.central_points = []
        self.critical_points = []

# class ObstacleSet2d(ObstacleSet):
#     def __init__(self, obstacles, boundary):
#         super().__init__(obstacles=obstacles, boundary=boundary)
    
#     def draw(self, ax):
#         x_points, y_points = self.boundary.exterior.xy
#         x_range = [min(x_points), max(x_points)]
#         y_range = [min(y_points), max(y_points)]
#         ax.set_xlim(x_range[0], x_range[1])
#         ax.set_ylim(y_range[0], y_range[1])
#         for obs in self.obstacles:
#             x,y = obs.exterior.xy
#             ax.plot(x,y, color='black')
    
# class TestSet(ObstacleSet2d):
#     def __init__(self):
#         obstacles = [
#             Polygon([
#                 [6, 6],
#                 [7, 6],
#                 [7, 7],
#                 [6, 7],            
#             ]),
#             Polygon([
#                 [2.5, -7.5],
#                 [5, -7.5],
#                 [5, 7.5],
#                 [2.5, 7.5],
#             ])
#         ]

#         x_range = [-10,10]
#         y_range = [-10,10]

#         boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])

#         super().__init__(obstacles=obstacles, boundary=boundary)

# class Shelves2d(ObstacleSet2d):
#     def __init__(self):
#         obstacles = []

#         x_range = [-10,10]
#         y_range = [-10,10]

#         boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])

#         super().__init__(obstacles=obstacles, boundary=boundary)

#         self.obstacles.append(create_rectangle_geometry(x_loc=3.5, y_loc=2.5, x_width=4, y_length=1))
#         self.obstacles.append(create_rectangle_geometry(x_loc=5.0, y_loc=0.5, x_width=1, y_length=3))
#         self.obstacles.append(create_rectangle_geometry(x_loc=3.5, y_loc=-1.5, x_width=4, y_length=1))

# class ParkingSpace(ObstacleSet2d):
#     def __init__(self):
#         obstacles = []
#         x_range = [-15,15]
#         y_range = [-15,15]
#         boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])
#         super().__init__(obstacles=obstacles, boundary=boundary)

#         self.obstacles.extend(self.create_parking_space(space_width=5))
#         self.obstacles.extend(self.create_parking_space(x_loc=-7.5, y_loc=-7.5, space_width=5))
    
#     def create_parking_space(self, x_loc=0, y_loc=0, space_width=5):
#         line_width = 0.5
#         line_height = 6
#         obs = [
#             Polygon([
#                 [x_loc, y_loc],
#                 [x_loc, y_loc+line_height],
#                 [x_loc+line_width, y_loc+line_height],
#                 [x_loc+line_width, y_loc],            
#             ]),
#             Polygon([
#                 [x_loc+space_width+line_width, y_loc],
#                 [x_loc+space_width+line_width, y_loc+line_height],
#                 [x_loc+space_width+line_width*2, y_loc+line_height],
#                 [x_loc+space_width+line_width*2, y_loc],            
#             ]),
#             Polygon([ # Horizontal Bar
#                 [x_loc+line_width, y_loc+line_height-line_width],
#                 [x_loc+line_width, y_loc+line_height],
#                 [x_loc+space_width+line_width, y_loc+line_height],
#                 [x_loc+space_width+line_width, y_loc+line_height-line_width],            
#             ]),
#         ]
#         x_center = (2 * x_loc + space_width + line_width*2) / 2
#         y_center = (2 * y_loc + line_height - line_width) / 2

#         sample_radius = 2
#         space_samples = np.array([x_center, y_center]) + (np.random.normal(size=(1000, 2)) * sample_radius)

#         self.central_points.append(np.array([x_center, y_center]))
#         self.critical_points.extend(space_samples)

#         return obs
        
# class RandomSamplePassage(ObstacleSet2d):
#     def __init__(self, num_walls=3, wall_width=1, gap_width=1):
#         obstacles = []
#         boundary = []

#         x_range = [0,(10 * (num_walls+1))]
#         y_range = [0,10]

#         # for x in x_range:
#             # for y in reversed(y_range):
#                 # boundary.append([x, y])
#         boundary = Polygon([[0, 0],
#                             [0, 10],
#                             [x_range[1], 10],
#                             [x_range[1], 0]])
        
#         for i in range(num_walls):
#             x_low = (10 * (i+1)) - wall_width/2
#             x_high = (10 * (i+1)) + wall_width/2
#             gap_y_loc = np.random.random() * (y_range[1] - y_range[0] - gap_width) + y_range[0] + gap_width/2

#             y_low = gap_y_loc - gap_width/2
#             y_high = gap_y_loc + gap_width/2
            
#             obs = Polygon([[x_low, y_range[0]],
#                            [x_low, y_low],
#                            [x_high, y_low],
#                            [x_high, y_range[0]]])
#             obstacles.append(obs)
            
#             obs = Polygon([[x_low, y_range[1]],
#                            [x_low, y_high],
#                            [x_high, y_high],
#                            [x_high, y_range[1]]])
#             obstacles.append(obs)

#         super().__init__(obstacles=obstacles, boundary=boundary)

# class CentralObstacle(ObstacleSet2d):
#     def __init__(self):
        
#         x_range = [0,10]
#         y_range = [0,10]

#         obstacles = []
#         # obs = create_rectangle_geometry(5,5,5.1,5)
#         obs = create_rectangle_geometry(5,5,2,2)
#         # obs = create_rectangle_geometry(5,5,0.01,0.01)
#         obstacles.append(obs)

#         boundary = Polygon([[0, 0],
#                             [0, 10],
#                             [x_range[1], 10],
#                             [x_range[1], 0]])

#         super().__init__(obstacles=obstacles, boundary=boundary)

# class BiasedPassage(ObstacleSet2d):
#     def __init__(self, num_walls=1, bias=0.5, main_wall_width=2, sup_wall_width=1, gap_width=1):
#         assert (num_walls > 0)

#         obstacles = []
#         boundary = []

#         x_range = [0,(10 * (num_walls+1))]
#         y_range = [0,10]

#         boundary = Polygon([[0, 0],
#                             [0, 10],
#                             [x_range[1], 10],
#                             [x_range[1], 0]])

#         for i in range(num_walls):
#             x_low = (10 * (i+1)) - main_wall_width/2
#             x_high = (10 * (i+1)) + main_wall_width/2

#             y_low = gap_width
#             y_high = y_range[1] - gap_width
            
#             obs = Polygon([[x_low, y_high],
#                            [x_low, y_low],
#                            [x_high, y_low],
#                            [x_high, y_high]])
#             obstacles.append(obs)

#             sup_x_low = (10 * (i+1)) - sup_wall_width/2
#             sup_x_high = (10 * (i+1)) + sup_wall_width/2

#             if np.random.random() < bias:
#                 sup_y_low = y_range[0]
#                 sup_y_high = y_range[0] + gap_width
#             else:
#                 sup_y_low = y_range[1] - gap_width
#                 sup_y_high = y_range[1]
            
#             obs = Polygon([[sup_x_low, sup_y_low],
#                            [sup_x_low, sup_y_high],
#                            [sup_x_high, sup_y_high],
#                            [sup_x_high, sup_y_low]])
#             obstacles.append(obs)

#         super().__init__(obstacles=obstacles, boundary=boundary)

# class WeavingPassage(ObstacleSet2d):
#     def __init__(self):
        
#         x_range = [0,10]
#         y_range = [0,10]

#         boundary = Polygon([[0, 0],
#                             [0, 10],
#                             [x_range[1], 10],
#                             [x_range[1], 0]])
        
#         obstacles = []

#         num_blocks = 9

#         for i in range(num_blocks):
#             x_loc = 3 if i % 2 == 0 else 7
#             obstacles.append(create_rectangle_geometry(x_loc, i+1, 7, 0.9))

#         super().__init__(obstacles=obstacles, boundary=boundary)

# class NonRegularPolygonObst(ObstacleSet2d):
#     def __init__(self):
#         obstacles = []

#         obs = Polygon([
#             (0,0),
#             (2,1),
#             (1.6,3),
#             (1.3,3.5),
#             (1.2,3),
#             (-0.1,1.3)
#         ])

#         obstacles.append(obs)

#         x_range = [-10,10]
#         y_range = [-10,10]

#         boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])
#         super().__init__(obstacles=obstacles, boundary=boundary)

# class Cubicles(ObstacleSet2d):
#     def __init__(self):

#         self.obstacles = []
#         self.central_points = []
#         self.critical_points = []

#         x_range = [0,45]
#         y_range = [0,50]

#         self.x_range = x_range
#         self.y_range = y_range

#         boundary = Polygon([(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])])
#         self.create_cubicle_sets()
        

#         super().__init__(obstacles=self.obstacles, boundary=boundary)

#     def create_single_cubicle(self, x_loc=0, y_loc=0, space_width=5, bottom_open=False):
#         line_width = 0.5
#         line_height = 6

#         horizontal_obstacle = None
        
#         if bottom_open:
#             horizontal_obstacle = Polygon([ # Horizontal Bar
#                     [x_loc+line_width, y_loc+line_height-line_width],
#                     [x_loc+line_width, y_loc+line_height],
#                     [x_loc+space_width+line_width, y_loc+line_height],
#                     [x_loc+space_width+line_width, y_loc+line_height-line_width],            
#                 ])
#         else:
#             horizontal_obstacle = Polygon([ # Horizontal Bar
#                     [x_loc+line_width, y_loc+line_width],
#                     [x_loc+line_width, y_loc],
#                     [x_loc+space_width+line_width, y_loc],
#                     [x_loc+space_width+line_width, y_loc+line_width],
#                 ])

#         obs = [
#             Polygon([
#                 [x_loc, y_loc],
#                 [x_loc, y_loc+line_height],
#                 [x_loc+line_width, y_loc+line_height],
#                 [x_loc+line_width, y_loc],            
#             ]),
#             Polygon([
#                 [x_loc+space_width+line_width, y_loc],
#                 [x_loc+space_width+line_width, y_loc+line_height],
#                 [x_loc+space_width+line_width*2, y_loc+line_height],
#                 [x_loc+space_width+line_width*2, y_loc],            
#             ]),
#             horizontal_obstacle
#         ]
#         x_center = (2 * x_loc + space_width + line_width*2) / 2
#         y_center = (2 * y_loc + line_height - line_width) / 2

#         sample_radius = 2
#         space_samples = np.array([x_center, y_center]) + (np.random.normal(size=(1000, 2)) * sample_radius)

#         self.central_points.append(np.array([x_center, y_center]))
#         self.critical_points.extend(space_samples)

#         return obs
    
#     def create_single_cubicle_set(self, x_loc, y_loc, space_width, num_spaces, bottom_open=False):
#         for i in range(num_spaces):
#             self.obstacles.extend(self.create_single_cubicle((i * 5) + x_loc, y_loc, space_width, bottom_open))

#     def get_x_loc_for_set(self, set_num, num_cubicles_per_set, cubicle_width):
#         width_of_cubicle_set = ((cubicle_width + 1) * num_cubicles_per_set)
#         return (width_of_cubicle_set * set_num) + 5 + (5 * set_num)

#     def create_cubicle_sets(self, num_cubicles_per_set=4, cubicle_width=4):
        
#         i = 0
#         set_i_x_loc_vert = self.get_x_loc_for_set(i, num_cubicles_per_set, cubicle_width)

#         while set_i_x_loc_vert < self.x_range[1]:
#             self.create_single_cubicle_set(set_i_x_loc_vert, 7, cubicle_width, num_cubicles_per_set, np.random.randint(0, 2))
#             self.create_single_cubicle_set(set_i_x_loc_vert, 21, cubicle_width, num_cubicles_per_set, np.random.randint(0, 2))
#             self.create_single_cubicle_set(set_i_x_loc_vert, 35, cubicle_width, num_cubicles_per_set, np.random.randint(0, 2))

#             i += 1
#             set_i_x_loc_vert = self.get_x_loc_for_set(i, num_cubicles_per_set, cubicle_width)


if __name__ == '__main__':
    # obs_set = TestSet()
    # obs_set.draw(plt.gca())
    # plt.show()

    # obs_set = Shelves2d()
    obs_set = Cubicles()
    obs_set.draw(plt.gca())
    plt.show()