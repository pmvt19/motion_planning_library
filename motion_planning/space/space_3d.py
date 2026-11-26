import time

import numpy as np
import rerun as rr

from motion_planning.obstacle_sets import ObstacleSet3D
from motion_planning.space import RobotSpace


class RobotSpace3D(RobotSpace):
    def __init__(self):
        self.x_range = [-10, 10]
        self.y_range = [-10, 10]
        self.z_range = [-10, 10]

    def set_obstacles(self, obstacle_set: ObstacleSet3D):
        self.obstacles = obstacle_set.obstacles
        self.boundary = obstacle_set.boundary

        mid_x, mid_y, mid_z, x_len, y_len, z_len = self.boundary

        min_x = mid_x - x_len / 2
        max_x = mid_x + x_len / 2
        min_y = mid_y - y_len / 2
        max_y = mid_y + y_len / 2
        min_z = mid_z - z_len / 2
        max_z = mid_z + z_len / 2

        self.x_range = [min_x, max_x]
        self.y_range = [min_y, max_y]
        self.z_range = [min_z, max_z]

        # Some Spaces Don't have Spatial Dimensions
        # In this case, do not adjust the c space lower and upper bounds
        # As this would adjust the angular dimensions instead
        # Example: FixedArm
        # This assumes the first two dimensions are the X and Y spatial information 
        # respectively
        if self.angular_dims_start is None or self.angular_dims_start >= 3:
            self.c_space_lower_bound[0] = min_x
            self.c_space_upper_bound[0] = max_x

            self.c_space_lower_bound[1] = min_y
            self.c_space_upper_bound[1] = max_y

            self.c_space_lower_bound[2] = min_z
            self.c_space_upper_bound[2] = max_z

    def xyzwhl_to_ordered_vertices(aa_rect_prism):
        # aa_rect (x,y,z,xl,yl,zl)
        x, y, z, xl, yl, zl = aa_rect_prism
        vertices = np.array([
            [x-xl/2, y-yl/2, z-zl/2],
            [x+xl/2, y-yl/2, z-zl/2],
            [x+xl/2, y+yl/2, z-zl/2],
            [x-xl/2, y+yl/2, z-zl/2],
            [x-xl/2, y-yl/2, z+zl/2],
            [x+xl/2, y-yl/2, z+zl/2],
            [x+xl/2, y+yl/2, z+zl/2],
            [x-xl/2, y+yl/2, z+zl/2],
        ])
        return vertices

    def draw_environment(self, ax, method="rerun"):
        if method == "rerun":
            rr.init("3D Environment", spawn=True)
            # TODO: Remove this loop, we can do this in one shot
            for i, obst in enumerate(self.obstacles):
                centers = obst[:3]
                half_sizes = obst[3:] / 2
                rr.log(f"obsts_{i}", rr.Boxes3D(centers=centers, half_sizes=half_sizes, colors=[0, 0, 255], fill_mode=rr.components.FillMode.Solid))
        elif method == "matplotlib":
            raise NotImplementedError
            # ax.set_xlim(self.x_range[0], self.x_range[1])
            # ax.set_ylim(self.y_range[0], self.y_range[1])
            # ax.set_zlim(self.z_range[0], self.z_range[1])
            
            # for prism in self.obstacles:
            #     ordered_verts = xyzwhl_to_ordered_vertices(prism)
            #     for a,b in edges:
            #         point_a = ordered_verts[a]
            #         point_b = ordered_verts[b]
            #         ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]], color='blue')
        else:
            raise ValueError("Drawing Method Not Supported")

    def animate_path(self, path, method='rerun'):
        for state in path:
            self.draw_environment(None)
            self.draw_state(None, state)
            time.sleep(0.1)