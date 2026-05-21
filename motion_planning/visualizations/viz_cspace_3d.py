import time

import matplotlib.pyplot as plt
import rerun as rr

from motion_planning.obstacle_sets import RandomSamplePassage
from motion_planning.space import PolygonalRobot
from motion_planning.visualizations.viz_cspace_2d import CSpaceVisualizer


class CSpaceVisualizer3D(CSpaceVisualizer):
    def __init__(self):
        super().__init__()

    def _animate_path_and_space(self, path, show_prev=True, frame_delay=0.01):
        rr.init("3D Configuration Space", spawn=True)
        start_time = time.time()
        for i, c in enumerate(path.path):
            rr.set_time("Time", duration=time.time() - start_time)
            rr.log(
                "Obstacle Points", rr.Points3D(self.obstacle_points, colors=[0, 0, 255])
            )
            rr.log("Robot Location", rr.Points3D(c.value, colors=[255, 0, 0]))

            self.env.draw_environment(plt.gca())
            self.env.draw_state(plt.gca(), c)

            plt.pause(frame_delay)  # Pause the Execution for frame_delay seconds
            plt.gca().cla()


class PolygonalRobotCSpaceVisualizer(CSpaceVisualizer3D):
    def __init__(self):
        super().__init__()

        self.env: PolygonalRobot = PolygonalRobot()
        self.env.set_obstacles(RandomSamplePassage(num_walls=1, gap_width=2))
        self._generate_obstacle_points()

    def run_interactive_space(self, tick_delay=0.01):
        # On the matplotlib side, you should be able to change the parameters 
        # of the polygonal robot and recompute the obstacle points
        raise NotImplementedError


if __name__ == "__main__":
    visualizer = PolygonalRobotCSpaceVisualizer()
    visualizer.run_visualized_search()
