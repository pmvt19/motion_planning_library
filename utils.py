import numpy as np
import matplotlib.pyplot as plt
from state import NumpyState, AngularNumpyState
from path import Path
from copy import deepcopy
from shapely import Polygon
import math
# from environments import Environment

def issue_warning(condition, statement, level):
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    if condition:
        print(f"{FAIL if level == 'fail' else WARNING}{statement}{ENDC}")

def smooth_path(env, path_obj : Path):
    path = deepcopy(path_obj.path) # Do not modify original path
    original_path_length = sum([np.linalg.norm(path[i].value - path[i+1].value) for i in range(len(path)-1)])
    i = 0
    while i < len(path)-1:
        j = len(path) - 1
        while i < j - 1:
            if env.is_valid_edge(path[i], path[j]):
                path = path[:(i+1)] + path[j:]
                break
            else:
                j -= 1
        i += 1
    smoothed_path_length = sum([np.linalg.norm(path[i].value - path[i+1].value) for i in range(len(path)-1)])
    print(f"Smoothed Path from Length {original_path_length} to Length {smoothed_path_length}")
    return Path(path=path)

def create_rectangle_geometry(x_loc, y_loc, x_width, y_length):
    shape = Polygon([[x_loc-x_width/2, y_loc-y_length/2], 
                        [x_loc-x_width/2, y_loc+y_length/2],
                        [x_loc+x_width/2, y_loc+y_length/2],
                        [x_loc+x_width/2, y_loc-y_length/2],])
    return shape

# To make general
# def interpolate_edge(start, end, delta):
#     x1, y1, theta1 = start
#     x2, y2, theta2 = end

#     edge_length = np.linalg.norm(end - start)
#     num_checks = int(edge_length / delta)
#     t_vals = np.linspace(0, 1, num_checks)  # Parameter t ∈ [0,1]
    
#     x_vals = (1 - t_vals) * x1 + t_vals * x2
#     y_vals = (1 - t_vals) * y1 + t_vals * y2
    
#     # Spherical interpolation of theta
#     cos_theta = (1 - t_vals) * np.cos(theta1) + t_vals * np.cos(theta2)
#     sin_theta = (1 - t_vals) * np.sin(theta1) + t_vals * np.sin(theta2)
#     theta_vals = np.arctan2(sin_theta, cos_theta)

#     return np.vstack((x_vals, y_vals, theta_vals)).T

# def interpolate_SE2_edge(start, end, delta):
#     x1, y1, theta1 = start
#     x2, y2, theta2 = end

#     edge_length = np.linalg.norm(end - start)
#     num_checks = int(edge_length / delta)
#     t_vals = np.linspace(0, 1, num_checks+2)  # Parameter t ∈ [0,1]
    
#     x_vals = (1 - t_vals) * x1 + t_vals * x2
#     y_vals = (1 - t_vals) * y1 + t_vals * y2
    
#     # Spherical interpolation of theta
#     cos_theta = (1 - t_vals) * np.cos(theta1) + t_vals * np.cos(theta2)
#     sin_theta = (1 - t_vals) * np.sin(theta1) + t_vals * np.sin(theta2)
#     theta_vals = np.arctan2(sin_theta, cos_theta)

#     return np.vstack((x_vals, y_vals, theta_vals)).T

# def interpolate_edge_mobile_arm(start, end, delta):
#     x1, y1, theta1_1, theta1_2, theta1_3 = start.value
#     x2, y2, theta2_1, theta2_2, theta2_3 = end.value

#     edge_length = np.linalg.norm(end.value - start.value)
#     num_checks = int(edge_length / delta)
#     t_vals = np.linspace(0, 1, num_checks)  # Parameter t ∈ [0,1]
    
#     x_vals = (1 - t_vals) * x1 + t_vals * x2
#     y_vals = (1 - t_vals) * y1 + t_vals * y2
    
#     # Spherical interpolation of theta
#     cos_theta1 = (1 - t_vals) * np.cos(theta1_1) + t_vals * np.cos(theta2_1)
#     sin_theta1 = (1 - t_vals) * np.sin(theta1_1) + t_vals * np.sin(theta2_1)
#     theta1_vals = np.arctan2(sin_theta1, cos_theta1)

#     cos_theta2 = (1 - t_vals) * np.cos(theta1_2) + t_vals * np.cos(theta2_2)
#     sin_theta2 = (1 - t_vals) * np.sin(theta1_2) + t_vals * np.sin(theta2_2)
#     theta2_vals = np.arctan2(sin_theta2, cos_theta2)

#     cos_theta3 = (1 - t_vals) * np.cos(theta1_3) + t_vals * np.cos(theta2_3)
#     sin_theta3 = (1 - t_vals) * np.sin(theta1_3) + t_vals * np.sin(theta2_3)
#     theta3_vals = np.arctan2(sin_theta3, cos_theta3)

#     # return np.vstack((x_vals, y_vals, theta_vals)).T
#     return np.vstack((x_vals, y_vals, theta1_vals, theta2_vals, theta3_vals)).T

# def interpolate_euclidean_edge(start : np.ndarray, end : np.ndarray, delta):
#     dir = (end - start) / np.linalg.norm(end - start)
#     interpolated_points = []
#     cur_node = start

#     edge_length = np.linalg.norm(end - start)
#     num_checks = int(edge_length / delta)

#     for i in range(num_checks):
#         interpolated_points.append(cur_node)
#         cur_node = cur_node + dir * delta
    
#     interpolated_points.append(end)

#     return interpolated_points

def calculate_edge_gradient(start_state : NumpyState, end_state : NumpyState):
    start = start_state.value
    end = end_state.value
    gradient = (end - start)

    if isinstance(start_state, AngularNumpyState):
        angular_dims_start = start_state.angular_dims_start
        gradient[angular_dims_start:] = np.arctan2(np.sin(gradient[angular_dims_start:]), np.cos(gradient[angular_dims_start:]))

    return gradient

def interpolate_edge(start : NumpyState, end : NumpyState, delta : float):

    gradient = calculate_edge_gradient(start, end)
    edge_length = np.linalg.norm(gradient)
    gradient /= edge_length
    
    num_checks = (math.ceil(edge_length / delta)) + 1
    edge_states_derivative = np.tile(gradient * delta, (num_checks, 1))
    edge_states_derivative[0] = np.zeros_like(start)
    edge_states = np.cumsum(edge_states_derivative, axis=0) + start.value
    edge_states[-1] = end.value

    if isinstance(start, AngularNumpyState):
        angular_dims_start = start.angular_dims_start
        edge_states[:, angular_dims_start:] = edge_states[:, angular_dims_start:] % (2*np.pi)
    return edge_states

def euclidean_distance(start, end):
    return np.linalg.norm(end-start)

def angular_distance(theta1, theta2): 
    two_pi = np.pi*2
    return np.minimum((theta1 - theta2) % two_pi, (theta2 - theta1) % two_pi)

def numpystate_distance(state1, state2):
    if isinstance(state1, AngularNumpyState) and isinstance(state2, AngularNumpyState) and (state1.angular_dims_start == state2.angular_dims_start):
        return euclidean_distance(state1.value[:state1.angular_dims_start], state2.value[:state2.angular_dims_start]) + \
                angular_distance(state1.value[state1.angular_dims_start:], state2.value[state2.angular_dims_start:])
    elif isinstance(state1, NumpyState) and isinstance(state2, NumpyState):
        return euclidean_distance(state1.value, state2.value)
    else:
        raise ValueError("Mismatched Types inputed or incorrect angular dims start")

# def interpolate_angular_edge(start, end, delta):
#     zero_mask = np.isclose(start, end)
#     if np.all(zero_mask):
#         vec = np.zeros_like(start)
#     else:
#         # if the angular distance does not match the absolute value of the difference then we wrap around...
#         dist = angular_distance(start, end)
#         diff = (end - start)
#         matched = 2*np.isclose(dist, np.abs(diff)) - 1.0
#         vec = matched * dist * diff/(np.abs(diff) + zero_mask*0.1**4)
#         vec /= np.linalg.norm(vec) # if we don't normalize then it splits every edge into equal number of pieces
    
#     # starting at c1 we move in increments of delta towards c2
#     pts = [np.copy(start)]
#     curr = np.copy(start) # NOTE: copy important to avoid changing original input
#     while True:
#         if np.linalg.norm(angular_distance(curr, end)) <= delta:
#             break
#         curr = (curr + vec*delta) % (2 * np.pi)
#         pts.append(np.copy(curr))
    
#     if not np.all(np.isclose(curr, end)):
#         pts.append(np.copy(end))
#     return np.array(pts)