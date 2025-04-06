import numpy as np
import pyvista as pv
from space import PolygonalRobot

if __name__ == '__main__':
    np.random.seed(0)
    env = PolygonalRobot()
    points = [env.sample_point() for _ in range(10000)]
    points = [point for point in points if not env.is_valid(point)]
    points =  np.array([point.value for point in points])
    print("Created Points")

    # NumPy array with shape (n_points, 3)
    point_cloud = pv.PolyData(points)
    mesh = point_cloud.reconstruct_surface(nbr_sz=100, sample_spacing=0.4)

    point_cloud.plot(eye_dome_lighting=True)
    mesh.plot(color='orange')