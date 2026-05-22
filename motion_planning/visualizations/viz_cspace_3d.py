import time

import matplotlib.pyplot as plt
import pygame
import rerun as rr
from matplotlib.widgets import Slider

from motion_planning.controller.xbox_controller import XboxController
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
        pygame.init()
        joysticks = []
        for i in range(0, pygame.joystick.get_count()):
            joysticks.append(pygame.joystick.Joystick(i))
            joysticks[-1].init()

        controller = XboxController(pygame)

        state = self.env.sample_valid_point()

        running = True

        fig, axs = plt.subplots(1, 1)
        fig.subplots_adjust(left=0.25, bottom=0.25)

        axlink1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        link1_slider = Slider(
            ax=axlink1,
            label="Robot Width",
            valmin=0.1,
            valmax=5,
            valinit=0.5,
            dragging=True,
        )

        axlink2 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        link2_slider = Slider(
            ax=axlink2,
            label="Robot Height",
            valmin=0.1,
            valmax=7,
            valinit=3,
            orientation="vertical",
            dragging=True,
        )

        def update_robot(val):
            self.env.robot_width = link1_slider.val
            self.env.robot_length = link2_slider.val
            self._generate_obstacle_points()

        link1_slider.on_changed(update_robot)
        link2_slider.on_changed(update_robot)

        # Disable events initially
        link1_slider.eventson = False
        link2_slider.eventson = False

        # On mouse release, trigger the event manually
        def on_release(event):
            if event.inaxes == link1_slider.ax:
                link1_slider.eventson = True  # Enable
                link1_slider._observers.process(
                    "changed", link1_slider.val
                )  # Manually trigger callbacks
                link1_slider.eventson = False  # Disable again immediately
            if event.inaxes == link2_slider.ax:
                link1_slider.eventson = True  # Enable
                link1_slider._observers.process(
                    "changed", link2_slider.val
                )  # Manually trigger callbacks
                link1_slider.eventson = False  # Disable again immediately

        fig.canvas.mpl_connect("button_release_event", on_release)

        rr.init("3D Configuration Space", spawn=True)
        start_time = time.time()
        while running:
            controller.update_state()
            controller_state = controller.get_contoller_state()
            x_dot = self.env.input_to_x_dot(controller_state)
            state = self.env.make_state(state.value + x_dot)

            # Plot Configuration Space (CSpace)
            rr.set_time("Time", duration=time.time() - start_time)
            rr.log(
                "Obstacle Points", rr.Points3D(self.obstacle_points, colors=[0, 0, 255])
            )
            rr.log("Robot Location", rr.Points3D(state.value, colors=[255, 0, 0]))

            axs.cla()

            self.env.draw_environment(axs)
            self.env.draw_state(axs, state)
            axs.set_title("Workspace")
            plt.pause(tick_delay)

            if controller_state[XboxController.XboxControls.LBUMPER]:
                running = False


if __name__ == "__main__":
    visualizer = PolygonalRobotCSpaceVisualizer()
    # visualizer.run_visualized_search()
    visualizer.run_interactive_space()
