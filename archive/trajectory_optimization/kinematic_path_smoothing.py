import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def smooth_path_trajectory_optimization(env, path):

    aabbs = []
    for obs in env.obstacles:
        coordinates_array = np.asarray(obs.exterior.coords)[:-1]
        x_min = np.min(coordinates_array[:, 0])
        x_max = np.max(coordinates_array[:, 0])

        y_min = np.min(coordinates_array[:, 1])
        y_max = np.max(coordinates_array[:, 1])

        aabbs.append((x_min, x_max, y_min, y_max))
        
    exit()
    # Time horizon
    N = len(path)  # number of discretization points

    # Define CasADi variables for state and control
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')

    # Create an empty list to store optimization variables
    state_traj = []

    # Set initial conditions and reference path (here we assume some predefined path)
    x_init, y_init = path[0].value
    x_final, y_final = path[-1].value

    # Create optimization variables
    opti = ca.Opti()

    # Create decision variables
    x_var = opti.variable(N)
    y_var = opti.variable(N)

    # Define cost function (minimize length)
    cost = 0
    # for k in range(N):
    #     cost += v_var[k]**2 + delta_var[k]**2  # simple cost on control effort

    # Add constraints for kinematic dynamics
    for k in range(N-1):
        # print(x_var[k])
        # print(np.array([x_var[k]]))
        # env.is_valid_edge(np.array([x_var[k], y_var[k]]), np.array([x_var[k+1], y_var[k+1]]))
        opti.subject_to(env.is_valid_edge(np.array([x_var[k], y_var[k]]), np.array([x_var[k+1], y_var[k+1]])))

    # Add initial conditions
    opti.subject_to(x_var[0] == x_init)
    opti.subject_to(y_var[0] == y_init)

    # Add boundary conditions to follow reference path
    # for k in range(N+1):
        # opti.subject_to(x_var[k] == reference_x[k])
        # opti.subject_to(y_var[k] == reference_y[k])

    opti.subject_to(x_var[N-1] == x_final)
    opti.subject_to(y_var[N-1] == y_final)

    # Solver options
    opti.solver('ipopt')

    # Solve the optimization problem
    sol = opti.solve()

    # Extract the optimized trajectory
    x_opt = sol.value(x_var)
    y_opt = sol.value(y_var)

    print(len(x_opt), len(path))
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_opt, y_opt, marker='o', label="Optimized Path")
    # plt.plot(reference_x, reference_y, '--', marker='o', label="Reference Path")
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.title('Optimized Trajectory vs. Reference Path')
    plt.grid()
    plt.show()