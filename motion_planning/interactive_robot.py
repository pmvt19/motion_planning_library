import time
import matplotlib.pyplot as plt
import pygame

from motion_planning.space import SkidSteerCar, DubinsCar
from motion_planning.controller.xbox_controller import XboxController
from motion_planning.obstacle_sets import ParkingSpace

def get_input(controller):
    controller.update_state()
    state = controller.get_contoller_state()
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

        plt.clf()
        env.draw_environment(plt.gca())
        env.draw_state(plt.gca(), state)
        plt.pause(0.01)

        if input[XboxController.XboxControls.LBUMPER]:
            print("Exiting")
            break