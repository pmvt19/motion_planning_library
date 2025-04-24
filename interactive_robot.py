from space import SkidSteerCar, DubinsCar
import time
import matplotlib.pyplot as plt
import pygame
from controller.xbox_controller import XboxController
from obstacle_sets import ParkingSpace
# import cv2

# class InteractiveRobot():
#     def __init__(self):
#         pass

#     def control(self):
#         raise NotImplementedError

# def read_inputs():
#     key = cv2.waitKey(0)
#     # print(key)
#     if key == ord('w'):
#         return [0, 0, 1, 0]
#     elif key == ord('a'):
#         return [1, 0, 0, 0]
#     elif key == ord('s'):
#         return [0, 0, 0, 1]
#     elif key == ord('d'):
#         return [0, 1, 0, 0]
#     return [0, 0, 0, 0]

# import select
# import sys

# def non_blocking_input():
#     i, o, e = select.select([sys.stdin], [], [], 0)
#     if i:
#         return sys.stdin.readline().strip()
#     return None

def get_input(controller):
    controller.update_state()
    state = controller.get_contoller_state()
    # state = state[6:10]
    return state

if __name__ == '__main__':

    pygame.init()
    joysticks = []
    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()
    
    
    controller = XboxController(pygame)

    # env = SkidSteerCar()
    env = DubinsCar()
    env.set_obstacles(ParkingSpace())
    state = env.sample_valid_point()

    while True: 
        input = get_input(controller)
        
        control = env.input_to_control(input)
        state = env.simulate_step(state, control)
        # print(input, state.value)

        plt.clf()
        env.draw_environment(plt.gca())
        env.draw_state(plt.gca(), state)
        plt.pause(0.01)
        # time.sleep(0.1)

        if input[XboxController.XboxControls.LBUMPER]:
            print("Exiting")
            break