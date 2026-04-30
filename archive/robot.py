
"""
This Robot Class file is not currently in use

The implementation details of this class are still being decided and will probably be used for an approximate collision checker.
"""

class Robot():
    def __init__(self, rectangles, segments, points):
        self.rectangles = rectangles
        self.segments = segments
        self.points = points
        assert ((len(self.rectangles) + len(self.segments) + len(self.points)) > 0), "Robot Must Include Some Geometry Representation"
    
class RobotApproximation():
    raise NotImplementedError

class BatchRobotApproximation():
    raise NotImplementedError