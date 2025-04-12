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


def batch_calculate_gradient(start_states : np.ndarray, end_states : np.ndarray, angular_dims_start):
    # start_states: (B, d), # end_states: (B, d)
    batch_gradient = end_states - start_states
    if angular_dims_start is not None:
        batch_gradient[:, angular_dims_start:] = np.arctan2(np.sin(batch_gradient[:, angular_dims_start:]), np.cos(batch_gradient[:, angular_dims_start:]))
    return batch_gradient

def batch_interpolate_edge(start_states : np.ndarray, end_states : np.ndarray, delta : float, angular_dims_start):
    # (B, d), # (B, d)
    B, d  = start_states.shape
    gradients = (end_states - start_states)
    gradients = batch_calculate_gradient(start_states, end_states, angular_dims_start)
    lengths = np.linalg.norm(gradients, axis=1)
    normalized_gradients = gradients / lengths.reshape(-1, 1) # TODO: Probably need to reshape # (B, d)

    num_steps = np.ceil((lengths / delta) + 1).astype(np.int32)
    max_steps = np.max(num_steps).astype(np.int32)

    normalized_gradients = normalized_gradients.reshape(-1, 1, d) # Reshape normalized vectors for repeating function
    edge_states_derivative = np.repeat(normalized_gradients * delta, (max_steps), axis=1) # (B, max_steps, d)
    edge_states_derivative[:,0,:] = 0
    edge_states = np.cumsum(edge_states_derivative, axis=1) + start_states.reshape(-1, 1, d) # (B, max_steps, d) + (B,1,d)
    edge_states[np.arange(B), (num_steps-1), :] = end_states
    return edge_states, num_steps

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

def interpolate_path(path : Path, delta : float):
    interpolated_path = []
    for i in range(len(path)-1):
        interpolated_path.extend(interpolate_edge(path[i], path[i+1], delta))
    return Path(interpolated_path)
