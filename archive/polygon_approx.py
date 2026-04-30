from space import RobotSpace, PlanarMobileArm, PolygonalRobot
from obstacle_sets import TestSet, NonRegularPolygonObst

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

def lines_to_point_dist(lines, point):
    # lines : (B, 2, 2) (Num Lines, Start/End, XY)
    # Point : (2,)

    numerator = (lines[:, 1, 1] - lines[:, 0, 1]) * point[0] - (lines[:, 1, 0] - lines[:, 0, 0]) * point[1] + lines[:, 1, 0] * lines[:, 0, 1] - lines[:, 1, 1] * lines[:, 0, 0]
    numerator = np.abs(numerator)

    line_lengths = np.linalg.norm(lines[:, 1] - lines[:, 0])

    dists = numerator / line_lengths

    return dists


def line_to_point_dist(p1, p2, point):
    # lines : (B, 2, 2) (Num Lines, Start/End, XY)
    # Point : (2,)

    numerator = abs((p2[1]-p1[1])*point[0] - (p2[0]-p1[0]*point[1] + p2[0]*p1[1] - p2[1]*p1[0]))
    length = np.linalg.norm((p1-p2))

    dist = numerator / length
    return dist
    



if __name__ == "__main__":
    np.random.seed(0)

    env = PolygonalRobot()
    # env.set_obstacles(TestSet())

    env.set_obstacles(NonRegularPolygonObst())

    coords = env.obstacles[0].exterior.coords
    coords = np.array(coords)[:-1, :]

    print(coords)
    centroid = np.mean(coords, axis=0)

    
    # for coord in coords[:-1]:
        # print(coord)

    env.draw_environment(plt.gca())
    # plt.scatter(centroid[0], centroid[1])

    # for coord in coords:
    #     plt.plot([centroid[0], coord[0]], [centroid[1], coord[1]])

    # plt.show()

    # lines = np.hstack((coords[:-1, :], coords[1:, :]))
    lines = np.concatenate((coords[:-1, :].reshape(-1, 1, 2), coords[1:, :].reshape(-1, 1, 2)), axis=1)
    # lines = lines.transpose(0, 2, 1)
    lines = lines[0:1]
    print(lines)
    print(lines.shape)
    print(f"Centroid: {centroid}")
    print(lines_to_point_dist(lines, centroid))

    print(line_to_point_dist(coords[0], coords[1], centroid))
    
    plt.scatter(centroid[0], centroid[1])
    for coord in coords:
        plt.plot([centroid[0], coord[0]], [centroid[1], coord[1]])
    # plt.scatter(centroid[0], centroid[1], s=1)

    patch_list = [patches.Circle(centroid, 0.4472135954999578)]
    patch_collection = PatchCollection(patch_list, color='red')
    ax = plt.gca()
    ax.add_collection(patch_collection)

    plt.show()