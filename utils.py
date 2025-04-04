import numpy as np
import matplotlib.pyplot as plt
from state import NumpyState, AngularNumpyState
from path import Path
from copy import deepcopy
# from environments import Environment

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

# To make general
def interpolate_edge(start, end, delta):
    x1, y1, theta1 = start
    x2, y2, theta2 = end

    edge_length = np.linalg.norm(end - start)
    num_checks = int(edge_length / delta)
    t_vals = np.linspace(0, 1, num_checks)  # Parameter t ∈ [0,1]
    
    x_vals = (1 - t_vals) * x1 + t_vals * x2
    y_vals = (1 - t_vals) * y1 + t_vals * y2
    
    # Spherical interpolation of theta
    cos_theta = (1 - t_vals) * np.cos(theta1) + t_vals * np.cos(theta2)
    sin_theta = (1 - t_vals) * np.sin(theta1) + t_vals * np.sin(theta2)
    theta_vals = np.arctan2(sin_theta, cos_theta)

    return np.vstack((x_vals, y_vals, theta_vals)).T
            
def interpolate_euclidean_edge(start : np.ndarray, end : np.ndarray, delta):
    dir = (end - start) / np.linalg.norm(end - start)
    interpolated_points = []
    cur_node = start

    edge_length = np.linalg.norm(end - start)
    num_checks = int(edge_length / delta)

    for i in range(num_checks):
        interpolated_points.append(cur_node)
        cur_node = cur_node + dir * delta
    
    interpolated_points.append(end)

    return interpolated_points

def euclidean_distance(start, end):
    return np.linalg.norm(end-start)

def angular_distance(theta1, theta2): 
    two_pi = np.pi*2
    return np.minimum((theta1 - theta2) % two_pi, (theta2 - theta1) % two_pi)

def numpystate_distance(state1, state2):
    if isinstance(state1, NumpyState) and isinstance(state2, NumpyState):
        return euclidean_distance(state1.value, state2.value)
    elif isinstance(state1, AngularNumpyState) and isinstance(state2, AngularNumpyState) and (state1.angular_dims_start == state2.angular_dims_start):
        return euclidean_distance(state1.value[state1.angular_dims_start], state2.value[state2.angular_dims_start]) + angular_distance
    else:
        raise ValueError("Mismatched Types inputed or incorrect angular dims start")

def interpolate_angular_edge(start, end, delta):
    zero_mask = np.isclose(start, end)
    if np.all(zero_mask):
        vec = np.zeros_like(start)
    else:
        # if the angular distance does not match the absolute value of the difference then we wrap around...
        dist = angular_distance(start, end)
        diff = (end - start)
        matched = 2*np.isclose(dist, np.abs(diff)) - 1.0
        vec = matched * dist * diff/(np.abs(diff) + zero_mask*0.1**4)
        vec /= np.linalg.norm(vec) # if we don't normalize then it splits every edge into equal number of pieces
    
    # starting at c1 we move in increments of delta towards c2
    pts = [np.copy(start)]
    curr = np.copy(start) # NOTE: copy important to avoid changing original input
    while True:
        if np.linalg.norm(angular_distance(curr, end)) <= delta:
            break
        curr = (curr + vec*delta) % (2 * np.pi)
        pts.append(np.copy(curr))
    
    if not np.all(np.isclose(curr, end)):
        pts.append(np.copy(end))
    return np.array(pts)