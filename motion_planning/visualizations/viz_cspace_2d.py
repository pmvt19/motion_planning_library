import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.widgets import Slider

from motion_planning.controller.xbox_controller import XboxController
from motion_planning.obstacle_sets import ParkingSpace, Shelves2d
from motion_planning.search import RRT
from motion_planning.space import ApproximationSpace, DiscRobot, FixedArm, RobotSpace
from motion_planning.utils import interpolate_path, smooth_path

# Interactive element
# - Shows workspace and cspace states as one uses the controller to manipulate the arm (Need to modify Holonomic Robots Class to make this work)
# - Potentially allow the user to change properties of the robot like arm lengths (Specific to this file)

# Search element
# - Shows workspace and cspace executing a planned rrt path (Done)
# - Add Colors to CSpace side that shows the progression of the path (Done)

#                   CSpaceVisualizer
# FixedArmCSpaceVisualizer & DiscRobotCSpaceVisualizer


class CSpaceVisualizer:
    def __init__(self):
        self.env: RobotSpace = None
        self.obstacle_points: np.ndarray = None

    def run_visualized_search(self, start=None, target=None):
        if start is None or target is None:
            print(
                "Either Start or Target is not specified."
                "Randomly Sampling Start and Target Configurations"
            )
            start = self.env.sample_valid_point()
            target = self.env.sample_valid_point()

        rrt = RRT(self.env)
        path = rrt.search(start, target, max_steps=5000)
        print(f"Path Length: {len(path)}")

        path = smooth_path(self.env, path)  # Should be optional
        path = interpolate_path(path, self.env, 0.05)
        self._animate_path_and_space(path, show_prev=False)

    def run_interactive_space(self):
        raise NotImplementedError

    def _generate_obstacle_points(self, sample_size=100000):
        approx_env = ApproximationSpace(self.env, batch_size=1000)
        start_time = time.time()
        points = np.array([approx_env.sample_point().value for _ in range(sample_size)])
        end_time = time.time()
        print(f"Time to Sample Points: {end_time - start_time}")

        start_time = time.time()
        point_validities = approx_env.batch_is_valid(points)
        self.obstacle_points = points[(point_validities == False)]
        end_time = time.time()
        print(f"Time to Validate Points: {end_time - start_time}")

    def _animate_path_and_space(self, path, show_prev=True, frame_delay=0.1):
        cmap = matplotlib.colormaps["viridis"]
        colors = [cmap(i / len(path)) for i in range(len(path))]

        fig, axs = plt.subplots(1, 2)
        axs[1].scatter(
            self.obstacle_points[:, 0], self.obstacle_points[:, 1], color="red"
        )

        start = path[0]
        target = path[-1]

        for i, c in enumerate(path.path):
            axs[0].cla()

            if not show_prev:
                axs[1].cla()
                axs[1].scatter(
                    self.obstacle_points[:, 0], self.obstacle_points[:, 1], color="red"
                )

            self.env.draw_environment(axs[0])
            self.env.draw_state(axs[0], c)
            axs[1].scatter(c.value[0], c.value[1], color=colors[i], marker="^")

            axs[1].scatter(start.value[0], start.value[1], color="green", marker="*")
            axs[1].scatter(target.value[0], target.value[1], color="red", marker="*")

            axs[0].set_aspect("equal")
            axs[1].set_aspect("equal")

            axs[0].set_title("Workspace")
            axs[1].set_title("Configuration Space")
            plt.pause(frame_delay)


class FixedArmCSpaceVisualizer(CSpaceVisualizer):
    def __init__(self):
        super().__init__()

        self.env: FixedArm = FixedArm()
        self.env.set_obstacles(Shelves2d())
        self.env.arm_link_lengths = np.array(
            [3, 3]
        )  # TODO HACK: DO NOT CHANGE ARM LENGTHS LIKE THIS
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
        fig, axs = plt.subplots(1, 2)
        fig.subplots_adjust(left=0.25, bottom=0.25)

        axlink1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        link1_slider = Slider(
            ax=axlink1,
            label="Link 1 Length",
            valmin=0.1,
            valmax=10,
            valinit=3,
            dragging=True,
        )

        axlink2 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        link2_slider = Slider(
            ax=axlink2,
            label="Link 2 Length",
            valmin=0.1,
            valmax=10,
            valinit=3,
            orientation="vertical",
            dragging=True,
        )

        def update_robot(val):
            self.env.arm_link_lengths = np.array([link1_slider.val, link2_slider.val])
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

        while running:
            controller.update_state()
            controller_state = controller.get_contoller_state()
            x_dot = self.env.input_to_x_dot(controller_state)
            state = self.env.make_state(state.value + x_dot)

            axs[0].cla()
            axs[1].cla()

            axs[1].scatter(
                self.obstacle_points[:, 0], self.obstacle_points[:, 1], color="red"
            )

            self.env.draw_environment(axs[0])
            self.env.draw_state(axs[0], state)
            axs[1].scatter(state.value[0], state.value[1], color="blue", marker="^")
            axs[0].set_aspect("equal")
            axs[1].set_aspect("equal")
            axs[0].set_title("Workspace")
            axs[1].set_title("Configuration Space")
            plt.pause(tick_delay)

            if controller_state[XboxController.XboxControls.LBUMPER]:
                running = False


# TODO: Fix Controller Integration
class DiscRobotCSpaceVisualizer(CSpaceVisualizer):
    def __init__(self):
        super().__init__()
        self.env = DiscRobot()
        self.env.set_obstacles(ParkingSpace())
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
        fig, axs = plt.subplots(1, 2)
        fig.subplots_adjust(left=0.25, bottom=0.25)

        axlink1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        link1_slider = Slider(
            ax=axlink1,
            label="Disc Radius",
            valmin=0.1,
            valmax=10,
            valinit=self.env.disc_radius,
            dragging=True,
        )

        def update_robot(val):
            self.env.disc_radius = link1_slider.val
            self._generate_obstacle_points()

        link1_slider.on_changed(update_robot)

        # Disable events initially
        link1_slider.eventson = False

        # On mouse release, trigger the event manually
        def on_release(event):
            if event.inaxes == link1_slider.ax:
                link1_slider.eventson = True  # Enable
                link1_slider._observers.process(
                    "changed", link1_slider.val
                )  # Manually trigger callbacks
                link1_slider.eventson = False  # Disable again immediately

        fig.canvas.mpl_connect("button_release_event", on_release)

        while running:
            controller.update_state()
            controller_state = controller.get_contoller_state()
            x_dot = self.env.input_to_x_dot(controller_state)
            state = self.env.make_state(state.value + x_dot)
            state = state

            axs[0].cla()
            axs[1].cla()

            axs[1].scatter(
                self.obstacle_points[:, 0], self.obstacle_points[:, 1], color="red"
            )

            self.env.draw_environment(axs[0])
            self.env.draw_state(axs[0], state)
            axs[1].scatter(state.value[0], state.value[1], color="blue", marker="^")
            axs[0].set_aspect("equal")
            axs[1].set_aspect("equal")

            axs[1].set_xlim(-15, 15)
            axs[1].set_ylim(-15, 15)
            axs[0].set_title("Workspace")
            axs[1].set_title("Configuration Space")
            plt.pause(tick_delay)

            if controller_state[XboxController.XboxControls.LBUMPER]:
                running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Configuration Space Visualizer",
        description=(
            "Allows you to visualize the sequence of states in Configuration "
            "Space of a Kinematic Robot Path. The path can either be generated "
            "via a traditional search algorithm or user generated via a controller"
        ),
    )

    parser.add_argument(
        "--task_type", default="search", choices=["interactive", "search"]
    )
    parser.add_argument(
        "--robot_type", default="FixedArm", choices=["FixedArm", "DiscRobot"]
    )

    args = parser.parse_args()

    task_type = args.task_type
    robot_type = args.robot_type

    np.random.seed(0)

    visualizer = None

    if robot_type == "FixedArm":
        visualizer = FixedArmCSpaceVisualizer()
    elif robot_type == "DiscRobot":
        visualizer = DiscRobotCSpaceVisualizer()

    if task_type == "search":
        visualizer.run_visualized_search()
    elif task_type == "interactive":
        visualizer.run_interactive_space()
