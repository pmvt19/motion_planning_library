from space import SkidSteerCar
import time
import matplotlib.pyplot as plt
# import pygame
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

def get_input():
    raise NotImplementedError

if __name__ == '__main__':
    
    

    env = SkidSteerCar()
    state = env.sample_valid_point()

    while True: 
        input = get_input()
        control = env.input_to_control(input)
        state = env.simulate_step(state, control)

        
        env.draw_environment(plt.gca())
        env.draw_state(plt.gca(), state)
        time.sleep(0.1)

    

    
    