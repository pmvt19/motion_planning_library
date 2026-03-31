import math
import time
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from motion_planning.space import RobotSpace, HolonomicRobot
from motion_planning.utils import numpystate_distance, interpolate_path
from motion_planning.state import NumpyState, AngularNumpyState
from motion_planning.prm import PRM
from motion_planning.obstacle_sets import ObstacleSet

def rect_prism_to_circles_x_short(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)

    radii = aa_rect_prism[3] / 2
    x, y, z, xl, yl, zl = aa_rect_prism

    segment_lengths = yl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    ys = [y-yl/2+radii]
    for i in range(num_points):
        ys.append(ys[-1] + delta)

    segment_lengths = zl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    zs = [z-zl/2+radii]
    for i in range(num_points):
        zs.append(zs[-1] + delta)

    ys = np.array(ys)
    zs = np.array(zs)

    output = np.array(np.meshgrid(ys,zs)).T.reshape(-1, 2)
    points = np.hstack((np.ones((output.shape[0],1))*x, output, np.ones((output.shape[0],1))*xl/2))

    return points

def rect_prism_to_circles_y_short(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)

    radii = aa_rect_prism[4] / 2
    x, y, z, xl, yl, zl = aa_rect_prism

    segment_lengths = xl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    xs = [x-xl/2+radii]
    for i in range(num_points):
        xs.append(xs[-1] + delta)

    segment_lengths = zl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    zs = [z-zl/2+radii]
    for i in range(num_points):
        zs.append(zs[-1] + delta)

    xs = np.array(xs)
    zs = np.array(zs)

    output = np.array(np.meshgrid(xs,zs)).T.reshape(-1, 2)
    points = np.hstack((output[:, 0].reshape(-1, 1), np.ones((output.shape[0],1))*y, output[:, 1].reshape(-1, 1), np.ones((output.shape[0],1))*yl/2))

    return points

def rect_prism_to_circles_z_short(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)

    radii = aa_rect_prism[5] / 2
    x, y, z, xl, yl, zl = aa_rect_prism

    segment_lengths = xl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    xs = [x-xl/2+radii]
    for i in range(num_points):
        xs.append(xs[-1] + delta)

    segment_lengths = yl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    ys = [y-yl/2+radii]
    for i in range(num_points):
        ys.append(ys[-1] + delta)

    xs = np.array(xs)
    ys = np.array(ys)

    output = np.array(np.meshgrid(xs,ys)).T.reshape(-1, 2)
    points = np.hstack((output, np.ones((output.shape[0],1))*z, np.ones((output.shape[0],1))*zl/2))

    return points

# TODO: Handle Overapproximations
def rect_prisms_to_circles(aa_rect_prisms):
    """
    aa_rect_prisms: (B, 6)
    """
    dim_sizes = aa_rect_prisms[:, 3:6]
    min_dims = np.argmin(dim_sizes, axis=1)

    x_min_dim_mask = min_dims==0
    y_min_dim_mask = min_dims==1
    z_min_dim_mask = min_dims==2

    x_min_prisms = aa_rect_prisms[x_min_dim_mask]
    y_min_prisms = aa_rect_prisms[y_min_dim_mask]
    z_min_prisms = aa_rect_prisms[z_min_dim_mask]

    circles = []
    for prism in x_min_prisms:
        circles.append(rect_prism_to_circles_x_short(prism).reshape(-1, 4))
    for prism in y_min_prisms:
        circles.append(rect_prism_to_circles_y_short(prism).reshape(-1, 4))
    for prism in z_min_prisms:
        circles.append(rect_prism_to_circles_z_short(prism).reshape(-1, 4))

    return np.concatenate(circles, axis=0)

# TODO: Handle Overapproximations
def cylinder_to_circles(cylinders, radius):
    """
    cylinders: (B, 2, 3)
    cyl_radii: (B, 1) or scaler
    """

    start_points = cylinders[:, 0, :] # (B, 3)
    end_points = cylinders[:, 1, :] # (B, 3)

    batch_rays = end_points - start_points
    segment_lengths = np.linalg.norm(batch_rays, axis=1).reshape(-1, 1) # (B,1)

    num_distinct_segment_lengths = len(np.unique(np.round(segment_lengths, 10)))
   
    batch_normalized_rays = batch_rays / segment_lengths # (B, 3)
    modified_segment_lengths = segment_lengths - (2*radius)

    num_circles_per_segment = np.ceil(np.round((modified_segment_lengths / (2*radius)), 10)).astype(np.int32)
    max_num_circles = math.ceil(np.max(num_circles_per_segment)) + 1

    circle_start_points = start_points + (batch_normalized_rays * radius)

    gaps = (modified_segment_lengths / num_circles_per_segment)

    batch_scaled_rays = (batch_normalized_rays * gaps.reshape(-1, 1)).reshape(-1, 1, 3)
    repeated_rays = np.repeat(batch_scaled_rays, (max_num_circles), axis=1)
    repeated_rays[:, 0, :] = 0

    trajectories = np.cumsum(repeated_rays, axis=1) + circle_start_points.reshape(-1, 1, 3)

    if isinstance(radius, float):
        shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
    elif isinstance(radius, np.ndarray):
        shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius.reshape(-1, 1, 1)

    circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)

    if num_distinct_segment_lengths == 1:
        circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 4)
    else:
        num_circles_per_segment = num_circles_per_segment.squeeze()
        circle_center_radius_pairs = np.vstack([circle_center_radius_pairs[i, :(num_circles+1)] for i, num_circles in enumerate(num_circles_per_segment)])

    return circle_center_radius_pairs

def circles_to_validity(obstacle_circles, robot_circles):
    """
    self.
    robot_circles: (B, N, 4)
    """
    B = robot_circles.shape[0]

    obst_xyz = obstacle_circles[:, :3]
    robot_xyz = robot_circles[:, :, :3]

    distance_mat = np.sqrt(np.sum(robot_xyz**2, axis=2, keepdims=True) + np.sum(obst_xyz**2, axis=1, keepdims=True).T + (-2 * (robot_xyz @ obst_xyz.T)))

    min_dists = robot_circles[:, :, 2].reshape(B, -1, 1) + obstacle_circles[:, 2].reshape(1, 1, -1)

    validity_mask = distance_mat > min_dists
    validity_mask = validity_mask.reshape(B, -1)
    validities = np.all(validity_mask, axis=1)
    return validities

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

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def visualize(prisms, edges, circles):
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(vertices),
    #     lines=o3d.utility.Vector2iVector(edges),
    # )

    line_sets = []
    for prism in prisms:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(xyzwhl_to_ordered_vertices(prism)),
            lines=o3d.utility.Vector2iVector(edges),
        )
        line_sets.append(line_set)

    mesh_circles = []
    for x,y,z,r in circles:

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)

        # 2. Define the new center coordinates
        new_center = np.array([x,y,z])

        # 3. Translate the sphere to the new center
        sphere.translate(new_center, relative=False)

        # 4. Compute vertex normals for proper shading
        sphere.compute_vertex_normals()
        mesh_circles.append(sphere)

    # 5. Visualize the sphere
    o3d.visualization.draw_geometries(mesh_circles+line_sets)

    # o3d.visualization.draw_geometries([line_set], zoom=0.8)

def viz_cylinder():
    radius = 1.0
    height = 2.0
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)

    # Define the target location
    target_location = np.array([5.0, 2.0, 1.0])

    # Translate the cylinder to the target location
    cylinder.translate(target_location)
    # cylinder.compute_vertex_normals()

    # Optional: Add a coordinate frame for reference
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # Visualize the cylinder (and optional coordinate frame)
    o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(cylinder), mesh_frame], mesh_show_wireframe=True)

def viz_circles(circles):
    mesh_circles = []
    for x,y,z,r in circles:

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)

        # 2. Define the new center coordinates
        new_center = np.array([x,y,z])

        # 3. Translate the sphere to the new center
        sphere.translate(new_center, relative=False)

        # 4. Compute vertex normals for proper shading
        sphere.compute_vertex_normals()
        mesh_circles.append(sphere)

    # 5. Visualize the sphere
    o3d.visualization.draw_geometries(mesh_circles)

class RobotSpace3D(RobotSpace):
    def __init__(self):
        pass

    def set_obstacles(self, obstacle_set: ObstacleSet):
        self.obstacles = obstacle_set.obstacles
        # self.boundary = obstacle_set.boundary

        # x_points, y_points = self.boundary.exterior.xy
        # self.x_range = [min(x_points), max(x_points)]
        # self.y_range = [min(y_points), max(y_points)]


class ApproximationSpace3D(RobotSpace):
    def __init__(self, space : RobotSpace, batch_size=1000, do_overapproximation=False):
        self.space = space
        self.batch_size = batch_size
        self.do_overapproximation = do_overapproximation

        self.edge_validity_delta = 0.1
        self.angular_dims_start = None

        self.obstacle_circles = self.space_to_circles()

        self.num_collision_checks = 0
    
    def obstacles_to_3d_aabb(self, obstacles): # TODO: Needs to be implemented properly with representations of 3d obsts
        aabbs = []
        for obs in obstacles: # These representations should be different
            aabbs.append(obs)
        return np.array(aabbs)
    
    def space_to_circles(self):
        aabbs = self.obstacles_to_3d_aabb(self.space.obstacles)
        obst_circles = self.prisms_to_circles(aabbs)
        return obst_circles

    def dist(self, state1, state2):
        return self.space.dist(state1, state2)

    def states_to_circles(self, states):
        representations = self.space.batch_get_robot_representations(states)
        B, *_ = states.shape

        prism_circles = self.prisms_to_circles(representations['prisms']).reshape(B, -1, 4)
        cylinder_circles = self.cylinders_to_circles(representations['cylinders'], 
                                                     representations['cylinder_radii']).reshape(B, -1, 4)
        point_circles = self.points_to_circles(representations['points'], representations['points_radii']).reshape(B, -1, 4)
        # print(prism_circles.shape, cylinder_circles.shape, point_circles.shape)
        state_circles = np.concatenate((prism_circles, cylinder_circles, point_circles), axis=1)
        return state_circles

    def prisms_to_circles(self, aa_rect_prisms):
        """
        aa_rect_prisms: (B, 6)
        """

        if len(aa_rect_prisms) == 0:
            return np.empty((0, 4))
        
        dim_sizes = aa_rect_prisms[:, 3:6]
        min_dims = np.argmin(dim_sizes, axis=1)

        x_min_dim_mask = min_dims==0
        y_min_dim_mask = min_dims==1
        z_min_dim_mask = min_dims==2

        x_min_prisms = aa_rect_prisms[x_min_dim_mask]
        y_min_prisms = aa_rect_prisms[y_min_dim_mask]
        z_min_prisms = aa_rect_prisms[z_min_dim_mask]

        circles = []
        for prism in x_min_prisms:
            circles.append(rect_prism_to_circles_x_short(prism).reshape(-1, 4))
        for prism in y_min_prisms:
            circles.append(rect_prism_to_circles_y_short(prism).reshape(-1, 4))
        for prism in z_min_prisms:
            circles.append(rect_prism_to_circles_z_short(prism).reshape(-1, 4))

        return np.concatenate(circles, axis=0)

    def cylinders_to_circles(self, cylinders, radius):
        """
        cylinders: (Bm, m, 2, 3)
        cyl_radii: (Bm, 1) or scaler

        returns: (Bm, m, 4)
        """
        Bm, m, _, _ = cylinders.shape

        if len(cylinders) == 0:
            return np.empty((0, m, 4))
        
        cylinders = cylinders.reshape(Bm * m, 2, 3)

        start_points = cylinders[:, 0, :] # (B, 3)
        end_points = cylinders[:, 1, :] # (B, 3)

        batch_rays = end_points - start_points
        segment_lengths = np.linalg.norm(batch_rays, axis=1).reshape(-1, 1) # (B,1)

        num_distinct_segment_lengths = len(np.unique(np.round(segment_lengths, 10)))
    
        batch_normalized_rays = batch_rays / segment_lengths # (B, 3)
        modified_segment_lengths = segment_lengths - (2*radius)

        num_circles_per_segment = np.ceil(np.round((modified_segment_lengths / (2*radius)), 10)).astype(np.int32)
        max_num_circles = math.ceil(np.max(num_circles_per_segment)) + 1

        circle_start_points = start_points + (batch_normalized_rays * radius)

        gaps = (modified_segment_lengths / num_circles_per_segment)

        batch_scaled_rays = (batch_normalized_rays * gaps.reshape(-1, 1)).reshape(-1, 1, 3)
        repeated_rays = np.repeat(batch_scaled_rays, (max_num_circles), axis=1)
        repeated_rays[:, 0, :] = 0

        trajectories = np.cumsum(repeated_rays, axis=1) + circle_start_points.reshape(-1, 1, 3)

        if isinstance(radius, float):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius
        elif isinstance(radius, np.ndarray):
            shaped_radius = np.ones((trajectories.shape[0], trajectories.shape[1], 1)) * radius.reshape(-1, 1, 1)

        circle_center_radius_pairs = np.concatenate((trajectories, shaped_radius), axis=2)

        if num_distinct_segment_lengths == 1:
            circle_center_radius_pairs = circle_center_radius_pairs.reshape(-1, 4)
        else:
            num_circles_per_segment = num_circles_per_segment.squeeze()
            circle_center_radius_pairs = np.vstack([circle_center_radius_pairs[i, :(num_circles+1)] for i, num_circles in enumerate(num_circles_per_segment)])

        circle_center_radius_pairs = circle_center_radius_pairs.reshape(Bm, m, 4)
        return circle_center_radius_pairs

    def points_to_circles(self, points, radii):
        shaped_radii = np.ones((points.shape[0], 1)) * radii
        radius_circles = np.concatenate((points, shaped_radii), axis=1)
        return radius_circles

    def circles_to_validity(self, obstacle_circles, robot_circles):
        """
        self.
        robot_circles: (B, N, 4)
        """
        B = robot_circles.shape[0]

        obst_xyz = obstacle_circles[:, :3]
        robot_xyz = robot_circles[:, :, :3]

        distance_mat = np.sqrt(np.sum(robot_xyz**2, axis=2, keepdims=True) + np.sum(obst_xyz**2, axis=1, keepdims=True).T + (-2 * (robot_xyz @ obst_xyz.T)))

        min_dists = robot_circles[:, :, 3].reshape(B, -1, 1) + obstacle_circles[:, 3].reshape(1, 1, -1)

        validity_mask = distance_mat > min_dists
        # print(min_dists, distance_mat)
        validity_mask = validity_mask.reshape(B, -1)
        validities = np.all(validity_mask, axis=1)
        return validities

    def batch_is_valid(self, states):
        robot_circles = self.states_to_circles(states)
        B = robot_circles.shape[0]
        self.num_collision_checks += B
        stacked_validities = []
        num_batches = math.ceil(B / self.batch_size)
        for i in range(num_batches):
            idx_start = i * self.batch_size
            idx_end = min((i+1)*self.batch_size, B)
            validities = self.circles_to_validity(self.obstacle_circles, robot_circles[idx_start:idx_end])
            stacked_validities.append(validities)
        stacked_validities = np.hstack(stacked_validities)
        return stacked_validities

    def draw_state(self, ax, state, method='o3d'):
        pass

    def draw_environment(self, ax, state, method='o3d'):
        if method == 'o3d':
            mesh_circles = []
            for x,y,z,r in self.obstacle_circles:

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)

                # 2. Define the new center coordinates
                new_center = np.array([x,y,z])

                # 3. Translate the sphere to the new center
                sphere.translate(new_center, relative=False)

                # 4. Compute vertex normals for proper shading
                sphere.compute_vertex_normals()
                mesh_circles.append(sphere)

            # 5. Visualize the sphere
            o3d.visualization.draw_geometries(mesh_circles)
        elif method == 'rerun':
            rr.init("3D Environment", spawn=True)
            rr.log("Obstacle Spheres", rr.Points3D([self.obstacle_circles[:, :3]], colors=[0, 0, 255], radii=self.obstacle_circles[:, 3]))

    def draw_state_env(self, ax, state, method='o3d'):
        
        if method == 'o3d':
            mesh_circles = []
            for x,y,z,r in self.obstacle_circles:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
                # 2. Define the new center coordinates
                new_center = np.array([x,y,z])
                # 3. Translate the sphere to the new center
                sphere.translate(new_center, relative=False)
                # 4. Compute vertex normals for proper shading
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([0.0, 0.0, 1.0])
                mesh_circles.append(sphere)
            
            state_circles = self.states_to_circles(np.array(state.value).reshape(1, -1))[0]
            state_spheres = []
            # print("HERE", state_circles.shape)
            for x,y,z,r in state_circles:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
                # 2. Define the new center coordinates
                new_center = np.array([x,y,z])
                # 3. Translate the sphere to the new center
                sphere.translate(new_center, relative=False)
                # 4. Compute vertex normals for proper shading
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                state_spheres.append(sphere)
            
            # o3d.visualization.draw_geometries(mesh_circles + state_spheres)
            return mesh_circles, state_spheres
        elif method == 'rerun':
            rr.init("3D State & Environment", spawn=True)
            rr.log("Obstacle Spheres", rr.Points3D([self.obstacle_circles[:, :3]], colors=[0, 0, 255], radii=self.obstacle_circles[:, 3]))

            state_circles = self.states_to_circles(np.array(state.value).reshape(1, -1))[0]
            rr.log("State Spheres", rr.Points3D([state_circles[:, :3]], colors=[255, 0, 0], radii=state_circles[:, 3]))

    def animate_path(self, path, method='o3d'):
        if method == 'o3d':
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            state = path[0]
            mesh_circles, state_spheres = self.draw_state_env(None, state, None)
            # vis.add_geometry(mesh_circles + state_spheres)
            for geom in mesh_circles + state_spheres:
                vis.add_geometry(geom)
            vis.run()

            for state in path:
                print(state.value)
                # time.sleep(0.5)
                vis.clear_geometries()
                mesh_circles, state_spheres = self.draw_state_env(None, state, None)
                # vis.update_geometry(mesh_circles + state_spheres)
                for geom in mesh_circles + state_spheres:
                    # vis.update_geometry(geom)
                    vis.add_geometry(geom)
                vis.poll_events()
                vis.update_renderer()
                vis.run()
                time.sleep(0.01)
        elif method == 'rerun':
            for state in path:
                self.draw_state_env(None, state, method='rerun')
                time.sleep(0.1)


    def sample_point(self):
        return self.space.sample_point()
    
    def is_valid(self, state):
        return self.space.is_valid(state)

    def make_state(self, state):
        return self.space.make_state(state)

    def batch_sample_points_around_target(self, targets):
        raise NotImplementedError
    
class SphereRobot(HolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.1

        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.z_range = [-10,10]

        # self.theta_range = [0, 2*np.pi]
        # self.angular_dims_start = 2

        self.robot_radius = 0.5

        self.obstacles = []

        self.do_boundary_check = True

        ### HARD CODED ###
        aa_rect_prism1 = np.array([0,0,0,1,5,5])
        aa_rect_prism2 = np.array([2.5,2.5,0,5,1,5])
        aa_rect_prism3 = np.array([2.5,-2.5,0,5,1,5])
        aa_rect_prism4 = np.array([2.5,0,-2.5,5,5,1])
        aa_rect_prism5 = np.array([2.5,0,2.5,5,5,1])

        # prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3])
        prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3, aa_rect_prism4, aa_rect_prism5])
        self.obstacles = prisms
        ### HARD CODED ###

        self.num_collision_checks = 0
    
    def make_state(self, state):
        return NumpyState(value=state)

    def sample_point(self):
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        z = np.random.uniform(low=self.z_range[0], high=self.z_range[1])
        return self.make_state(np.array([x, y, z]))
    
    def generate_robot_representation(self, state):
        xCenter, yCenter, zCenter = self.get_state_value(state)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)
        # shift and scale sphere
        x = r*x + xCenter
        y = r*y + yCenter
        z = r*z + zCenter
        return (x,y,z)

    def dist(self, state1, state2):
        return numpystate_distance(state1, state2)

    def is_valid(self, state):
        raise NotImplementedError
    
    def draw_state(self, ax, state):
        cpx, cpy, cpz = self.generate_robot_representation(state)
        # ax.plot_surface(cpx, cpy, cpz, color="r")
        ax.plot_wireframe(cpx, cpy, cpz, color="r")

    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.set_zlim(self.z_range[0], self.z_range[1])
        
        for prism in prisms:
            ordered_verts = xyzwhl_to_ordered_vertices(prism)
            for a,b in edges:
                point_a = ordered_verts[a]
                point_b = ordered_verts[b]
                ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]], color='blue')

    def batch_get_robot_representations(self, states):
        return {
            'prisms' : np.empty((0, 6)),
            'cylinders' : np.empty((0, 0, 2, 3)), 
            'cylinder_radii' : 0.0,
            'points' : states, 
            'points_radii' : self.robot_radius
        }
    
    def batch_sample_points_around_target(self, targets):
        validities = self.batch_is_valid(targets)
        return targets[validities]

class UR5(HolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.1

        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.z_range = [-10,10]

        self.theta_range = [0, 2*np.pi]
        self.angular_dims_start = 0

        self.robot_radius = 0.5

        self.obstacles = []

        self.do_boundary_check = True

        ### HARD CODED ###
        aa_rect_prism1 = np.array([0,0,0,1,5,5])
        aa_rect_prism2 = np.array([2.5,2.5,0,5,1,5])
        aa_rect_prism3 = np.array([2.5,-2.5,0,5,1,5])
        aa_rect_prism4 = np.array([2.5,0,-2.5,5,5,1])
        aa_rect_prism5 = np.array([2.5,0,2.5,5,5,1])

        # prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3])
        prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3, aa_rect_prism4, aa_rect_prism5])
        self.obstacles = prisms
        ### HARD CODED ###

        self.num_collision_checks = 0
    
    def make_state(self, state):
        return AngularNumpyState(value=state, angular_dims_start=self.angular_dims_start)

    def sample_point(self):
        theta_0 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        theta_1 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        theta_2 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        return self.make_state(np.array([theta_0, theta_1, theta_2]))
    
    def generate_robot_representation(self, state):
        raise NotImplementedError

    def dist(self, state1, state2):
        return numpystate_distance(state1, state2)

    def is_valid(self, state):
        raise NotImplementedError
    
    def draw_state(self, ax, state):
        raise NotImplementedError

    def draw_environment(self, ax):
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.set_zlim(self.z_range[0], self.z_range[1])
        
        for prism in prisms:
            ordered_verts = xyzwhl_to_ordered_vertices(prism)
            for a,b in edges:
                point_a = ordered_verts[a]
                point_b = ordered_verts[b]
                ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]], color='blue')
    
    def forward_kinematics(self, state: NumpyState):
        theta1, theta2 = state.value
        # H_j1f_2_wf
        H1 = np.array([[np.cos(theta1), -np.sin(theta1), 0.0, 0.0],
                       [np.sin(theta1), np.cos(theta1), 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
        
        # H_j2f_2_j1f
        H2 = np.array([[np.cos(theta2), 0.0, np.sin(theta2), 1.0],
                       [0.0,            1.0,            0.0, 0.0],
                       [-np.sin(theta2), 0.0, np.cos(theta2), 1.0],
                       [0.0,             0.0,            0.0, 1.0]])
        
        homogenous_origin = np.array([0.0, 0.0, 0.0, 1.0])

        ee = H1 @ H2 @ homogenous_origin
        print(ee)
        print(H2 @ homogenous_origin)
    
    def batch_forward_kinematics(self, states: np.ndarray):
        # states: (N, m)
        # returns: (N, m, 2, 3)
        N, m = states.shape

        # H1 = np.array([[np.cos()]])

        # return np.empty((0, m, 2, 3))

        out1 = np.array([[[[0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0]],
                          [[1.0, 1.0, 1.0],
                           [0.0, 0.0, 2.0]]]])
        print(out1.shape)
        return out1

    def batch_get_robot_representations(self, states):

        cylinder_endpoints = self.batch_forward_kinematics(states)

        return {
            'prisms' : np.empty((0, 6)),
            'cylinders' : cylinder_endpoints, 
            'cylinder_radii' : 1.0,
            'points' : states, 
            'points_radii' : self.robot_radius
        }
    
    def batch_sample_points_around_target(self, targets):
        raise NotImplementedError

if __name__ == '__main__':

    center = (0,0,0)
    lengths = (2,2,2)

    edges = np.array([
        [0,1], # Bottom Face
        [1,2],
        [2,3],
        [3,0], # Bottom Face
        
        [4,5], # Top Face
        [5,6],
        [6,7],
        [7,4], # Top Face

        [0,4], # Mid Faces
        [1,5],
        [2,6],
        [3,7], # Mid Faces
    ])

    # vertices = np.array([
    #     [1,1,-1], # 0
    #     [1,-1,-1], # 1
    #     [-1,-1,-1], # 2
    #     [-1,1,-1], # 3 
    #     [1,1,1], # 4 
    #     [1,-1,1], # 5
    #     [-1,-1,1], # 6 
    #     [-1,1,1], # 7
    # ])

    # ax = plt.axes(projection='3d')
    # for a,b in edges:
    #     point_a = vertices[a]
    #     point_b = vertices[b]

    #     ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]])
    # plt.show()

    # aa_rect_prism1 = np.array([0,0,0,1,5,5])
    # aa_rect_prism2 = np.array([0,2.5,2.5,5,1,5])
    # aa_rect_prism3 = np.array([0,-2.5,2.5,5,1,5])

    ## Batch Commenting out START ##

    # aa_rect_prism1 = np.array([0,0,0,1,5,5])
    # aa_rect_prism2 = np.array([2.5,2.5,0,5,1,5])
    # aa_rect_prism3 = np.array([2.5,-2.5,0,5,1,5])

    # # prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3])
    # prisms = np.array([aa_rect_prism1, aa_rect_prism2, aa_rect_prism3])
    # circles = rect_prisms_to_circles(prisms)
    # # print(circles.shape)

    # ax = plt.axes(projection='3d')
    # for prism in prisms:
    #     ordered_verts = xyzwhl_to_ordered_vertices(prism)
    #     for a,b in edges:
    #         point_a = ordered_verts[a]
    #         point_b = ordered_verts[b]
    #         ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]])
    #     ax.scatter(circles[:, 0], circles[:, 1], circles[:, 2])
    #     # ax.set_box_aspect([[-5,5],[-5,5],[-5,5]])
    #     ax.set_box_aspect([1,1,1])
    #     amin = -11
    #     amax = 11
    #     ax.set_xlim(amin, amax)
    #     ax.set_ylim(amin, amax)
    #     ax.set_zlim(amin, amax)
    
    # for x,y,z,r in circles:
    #     # cpx, cpy, cpz = drawSphere(x,y,z,1*math.sqrt(3))
    #     cpx, cpy, cpz = drawSphere(x,y,z,r)
    #     # ax.plot_wireframe(cpx, cpy, cpz, color="r")
    #     ax.plot_surface(cpx, cpy, cpz, color="r")
    # plt.show()

    # visualize(prisms, edges, circles)
    # viz_cylinder()

    # end_points = np.array([[[0.0,0.0,0.0],
    #                        [3.8,0.0,0.0]]])
    # print(f"End Points: {end_points.shape}")
    # cirs = cylinder_to_circles(end_points, 0.3)
    # viz_circles(cirs)
    # print(circles.shape, cirs.shape)
    # print(circles_to_validity(circles, cirs.reshape(1, -1, 4)))

    ## Batch Commenting out END ##


    # aa_rect_prism = np.array([0,0,0,2,3,3.1])
    # aa_rect_prism = np.array([0,0,0,2,3.1,3])

    # aa_rect_prism = np.array([0,0,0,2,11.34,21.67])


    # aa_rect_prism = np.array([0,0,0,20.82,2,21.67])


    # aa_rect_prism = np.array([0,0,0,20.82,21.67,2])
    # ordered_verts = xyzwhl_to_ordered_vertices(aa_rect_prism)

    # ax = plt.axes(projection='3d')
    # for a,b in edges:
    #     point_a = ordered_verts[a]
    #     point_b = ordered_verts[b]
    #     ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]])
    # ax.scatter(circles[:, 0], circles[:, 1], circles[:, 2])
    # # ax.set_box_aspect([[-5,5],[-5,5],[-5,5]])
    # ax.set_box_aspect([1,1,1])
    # amin = -11
    # amax = 11
    # ax.set_xlim(amin, amax)
    # ax.set_ylim(amin, amax)
    # ax.set_zlim(amin, amax)

    # for x,y,z,r in circles:
    #     # cpx, cpy, cpz = drawSphere(x,y,z,1*math.sqrt(3))
    #     cpx, cpy, cpz = drawSphere(x,y,z,r)
    #     # ax.plot_wireframe(cpx, cpy, cpz, color="r")
    #     ax.plot_surface(cpx, cpy, cpz, color="r")

    # plt.show()

    ## Sphere Robot PRM Search START ##
    import time
    env = SphereRobot()
    env = ApproximationSpace3D(env)

    env.draw_environment(None, None, method='rerun')

    prm = PRM(env, num_samples=1000, num_neighbors=5, validate_edges=True)
    
    start_time = time.time()
    prm.create_graph()
    end_time = time.time()
    print(f"Time to create graph: {end_time - start_time}")
    # start, target = env.make_state(np.array([1.0,1.0,1.0])), env.make_state(np.array([5.0,5.0,5.0]))
    # start, target = env.make_state(np.array([1.0,1.0,1.0])), env.make_state(np.array([-2.0,1.0,1.0]))
    start, target = env.make_state(np.array([1.5,1.0,1.0])), env.make_state(np.array([-2.0,1.0,1.0]))

    start, target = env.make_state(np.array([2.5, 0.0, 0.0])), env.make_state(np.array([-2.5, 0.0, 0.0]))
    start_time = time.time()
    path = prm.search(start, target)
    # path = prm.search(target, start)
    end_time = time.time()
    print(f"Time to Search: {end_time - start_time}")
    print(path.path)

    print([state.value for state in path])
    path_states = np.array([state.value for state in path])
    print(env.batch_is_valid(path_states))
    path = interpolate_path(path, env, 0.1)
    path_states = np.array([state.value for state in path])
    print(env.batch_is_valid(path_states))
    env.animate_path(path, method='rerun')

    ## Sphere Robot PRM Search END ##

    # prm.draw(plt.gca())
    # plt.show()

    # plt.clf()
    # ax = plt.axes(projection='3d')
    # env.space.draw_environment(ax)
    # env.space.draw_state(ax, path[0])
    # plt.show()

    # plt.clf()
    # path = interpolate_path(path, env, 0.1)
    # for i in range(len(path)):
    #     ax = plt.axes(projection='3d')
    #     env.space.draw_environment(ax)
    #     env.space.draw_state(ax, path[i])
    #     plt.pause(0.1)

    env_base = UR5()
    env = ApproximationSpace3D(env_base)

    # env.draw_environment(None, None, method='rerun')

    state = env.make_state(np.array([0.0, 0.0]))

    env_base.forward_kinematics(state)
    state_circles = env.states_to_circles(np.array(state.value).reshape(1, -1))[0]
    print(state_circles)
