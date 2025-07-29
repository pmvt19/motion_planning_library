# pointcloud.txt is an array of points in the format:
# x1 y1 z1
# x2 y2 z2
# ...
# # This script reads the point cloud data from a file and generates a mesh using Open3D.
import open3d as o3d
import numpy as np

def load_point_cloud(file_path:str) -> o3d.geometry.PointCloud:
    """Load point cloud data from a text file."""
    points = np.loadtxt(file_path)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def compute_alpha_shape_mesh(pcd, alpha=0.05):
    """Compute mesh using alpha shape; returns a surface mesh ignoring internal points"""
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh

def smooth_mesh(mesh, iterations=10):
    """Apply Laplacian smoothing to mesh"""
    return mesh.filter_smooth_simple(number_of_iterations=iterations)

if __name__ == "__main__":
    # Load point cloud from file
    point_cloud = load_point_cloud("saves/point_cloud_new.txt")
    
    # Generate mesh with alpha shape
    alpha = 0.71  # adjust based on point density
    mesh = compute_alpha_shape_mesh(point_cloud, alpha)

    # Optional smoothing
    mesh = smooth_mesh(mesh, iterations=5)
    mesh.compute_triangle_normals()
    # Visualize
    o3d.visualization.draw_geometries([mesh])
    # save mesh to file 
    o3d.io.write_triangle_mesh("saves/output_mesh.ply", mesh)
    # save in mesh format
    
    print("Mesh saved to output_mesh.ply")



    # OLD BROKEN CODE:
    
    # np.savetxt('saves/point_cloud_new.txt', points)

    # # NumPy array with shape (n_points, 3)
    # point_cloud = pv.PolyData(points)
    # mesh = point_cloud.reconstruct_surface(nbr_sz=100, sample_spacing=0.4)

    # point_cloud.plot(eye_dome_lighting=True)
    # mesh.plot(color='orange')

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # # Estimate normals for the point cloud
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

    # # Apply Poisson Surface Reconstruction
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # # Visualize the result
    # o3d.visualization.draw_geometries([mesh])
    