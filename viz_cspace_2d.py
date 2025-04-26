import numpy as np
from space import FixedArm
from circle_approximation import ApproximationSpace
from obstacle_sets import TestSet, ParkingSpace
import matplotlib.pyplot as plt
import time
from utils import smooth_path, interpolate_path
import matplotlib
from matplotlib.widgets import Slider

import pygame
from controller.xbox_controller import XboxController


from rrt import RRT

# Interactive element 
# - Shows workspace and cspace states as one uses the controller to manipulate the arm (Need to modify Holonomic Robots Class to make this work)
# - Potentially allow the user to change properties of the robot like arm lengths (Specific to this file)

# Search element
# - Shows workspace and cspace executing a planned rrt path (Done)
# - Add Colors to CSpace side that shows the progression of the path (Done)

def generate_obstacle_points(env):
    # env = ApproximationSpace(env)
    start_time = time.time()
    points = np.array([env.sample_point().value for _ in range(1000)])
    end_time = time.time()
    print(f"Time to Sample Points: {end_time-start_time}")

    start_time = time.time()
    point_validities = env.batch_is_valid(points)
    obstacle_points = points[(point_validities == False)]
    end_time = time.time()
    print(f"Time to Validate Points: {end_time-start_time}")

    return obstacle_points

def animate_path_and_space(path, obstacle_points, show_prev=True, frame_delay=0.1):
    cmap = matplotlib.colormaps['viridis']
    colors = [cmap(i/len(path)) for i in range(len(path))]

    fig, axs = plt.subplots(1,2)
    axs[1].scatter(obstacle_points[:, 0], obstacle_points[:, 1], color='red')

    for i, c in enumerate(path.path):
        axs[0].cla()

        if not show_prev:
            axs[1].cla()
            axs[1].scatter(obstacle_points[:, 0], obstacle_points[:, 1], color='red')

        env.draw_environment(axs[0])
        env.draw_state(axs[0], c)
        axs[1].scatter(c.value[0], c.value[1], color=colors[i], marker='^')
        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')
        plt.pause(frame_delay)

def run_visualized_search(env, obstacle_points):
    start, target = env.sample_valid_point(), env.sample_valid_point()
    rrt = RRT(env)
    path = rrt.search(start, target, max_steps=1000)

    path = interpolate_path(path, env, 0.05)
    animate_path_and_space(path, obstacle_points, show_prev=False)



def run_interactive_space(env, obstacle_points, tick_delay=0.01):
    global my_obstacle_points
    my_obstacle_points = obstacle_points

    pygame.init()
    joysticks = []
    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()
    
    controller = XboxController(pygame)

    state = env.sample_valid_point()

    running = True
    fig, axs = plt.subplots(1,2)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    axlink1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    link1_slider = Slider(
        ax=axlink1,
        label='Link 1 Length',
        valmin=0.1,
        valmax=10,
        valinit=3,
        dragging=True
    )

    axlink2 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    link2_slider = Slider(
        ax=axlink2,
        label='Link 2 Length',
        valmin=0.1,
        valmax=10,
        valinit=3,
        orientation="vertical",
        dragging=True
    )

    
    def update_robot(val):
        env.arm_link_lengths = np.array([link1_slider.val, link2_slider.val])
        global my_obstacle_points
        my_obstacle_points = generate_obstacle_points(env)

    link1_slider.on_changed(update_robot)
    link2_slider.on_changed(update_robot)

    # Disable events initially
    link1_slider.eventson = False
    link2_slider.eventson = False

    # On mouse release, trigger the event manually
    def on_release(event):
        if event.inaxes == link1_slider.ax:
            link1_slider.eventson = True  # Enable
            link1_slider._observers.process('changed', link1_slider.val)  # Manually trigger callbacks
            link1_slider.eventson = False  # Disable again immediately
        if event.inaxes == link2_slider.ax:
            link1_slider.eventson = True  # Enable
            link1_slider._observers.process('changed', link2_slider.val)  # Manually trigger callbacks
            link1_slider.eventson = False  # Disable again immediately

    fig.canvas.mpl_connect('button_release_event', on_release)

    while running:
        controller.update_state()
        controller_state = controller.get_contoller_state()
        x_dot = env.input_to_x_dot(controller_state)
        state = env.make_state(state.value + x_dot)

        axs[0].cla()
        axs[1].cla()
        
        axs[1].scatter(my_obstacle_points[:, 0], obstacle_points[:, 1], color='red')

        env.draw_environment(axs[0])
        env.draw_state(axs[0], state)
        axs[1].scatter(state.value[0], state.value[1], color='blue', marker='^')
        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')
        plt.pause(tick_delay)

        if controller_state[XboxController.XboxControls.LBUMPER]:
            running = False

if __name__ == '__main__':

    # task_type = 'search' 
    task_type = 'interactive' # Will be made as an argument

    # np.random.seed(0)
    env = FixedArm()
    env.arm_link_lengths = np.array([3,3]) # HACK: DO NOT CHANGE ARM LENGTHS LIKE THIS
    env.set_obstacles(TestSet())

    obstacle_points = generate_obstacle_points(env)

    if task_type == "search":
        run_visualized_search(env, obstacle_points)
    elif task_type == "interactive":
        run_interactive_space(env, obstacle_points)