import numpy as np
import open3d as o3d

from motion_planning.space import PolygonalRobot
from motion_planning.obstacle_sets import CentralObstacle

def compute_alpha_shape_mesh(pcd, alpha=0.05):
    """Compute mesh using alpha shape; returns a surface mesh ignoring internal points"""
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh

def smooth_mesh(mesh, iterations=10):
    """Apply Laplacian smoothing to mesh"""
    return mesh.filter_smooth_simple(number_of_iterations=iterations)

if __name__ == '__main__':
    np.random.seed(0)
    env = PolygonalRobot()
    env.set_obstacles(CentralObstacle())
    points = [env.sample_point() for _ in range(10000)]
    points = [point for point in points if not env.is_valid(point)]
    points =  np.array([point.value for point in points])
    print("Created and Validated Points")

    # Create Open3d Point Cloud
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    alpha = 0.71  # adjust based on point density
    mesh = compute_alpha_shape_mesh(point_cloud, alpha)

    # Optional smoothing
    mesh = smooth_mesh(mesh, iterations=5)
    mesh.compute_triangle_normals()
    # Visualize
    o3d.visualization.draw_geometries([mesh])
