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