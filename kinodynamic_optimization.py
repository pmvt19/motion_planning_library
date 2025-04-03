import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from environments import CarParkingEnv
# Car parameters

def traj_opt_smoothing(env, path):

    reference_x = [p.value[0] for p in path]
    reference_y = [p.value[1] for p in path]
    reference_theta = [p.value[2] for p in path]
    # Time horizon
    dt = 0.1  # seconds
    N = len(path)-1  # number of discretization points

    # Define CasADi variables for state and control
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    theta = ca.MX.sym('theta')

    v = ca.MX.sym('v')  # velocity
    delta = ca.MX.sym('delta')  # steering angle

    # Kinematic model
    dx = v * ca.cos(theta)
    dy = v * ca.sin(theta)
    dtheta = delta

    # Define state and control vectors
    states = ca.vertcat(x, y, theta)
    controls = ca.vertcat(v, delta)

    # Create an empty list to store optimization variables
    state_traj = []
    control_traj = []

    # Set initial conditions and reference path (here we assume some predefined path)
    x_init = path[0].value[0]
    y_init = path[0].value[1]
    theta_init = path[0].value[2]
    v_init = 0.0  # Initial velocity
    delta_init = 0.0  # Initial angular velocity

    x_final = path[-1].value[0]
    y_final = path[-1].value[1]
    theta_final = path[-1].value[2]

    # Create optimization variables
    opti = ca.Opti()

    # Create decision variables
    x_var = opti.variable(N+1)
    y_var = opti.variable(N+1)
    theta_var = opti.variable(N+1)
    v_var = opti.variable(N)
    delta_var = opti.variable(N)

    # Define cost function (minimize control effort)
    cost = 0
    for k in range(N):
        cost += v_var[k]**2 + delta_var[k]**2  # simple cost on control effort

    # Add constraints for kinematic dynamics
    
    for k in range(N):
        # dt = T / N
        opti.subject_to(x_var[k+1] == x_var[k] + dt * v_var[k] * ca.cos(theta_var[k]))
        opti.subject_to(y_var[k+1] == y_var[k] + dt * v_var[k] * ca.sin(theta_var[k]))
        opti.subject_to(theta_var[k+1] == theta_var[k] + dt * delta_var[k])

    M = 50
    for (xmin, xmax, ymin, ymax) in aabbs:
        for k in range(N):  # Apply at each time step
            s = opti.variable(4)  # Slack variables for each OR condition
            opti.subject_to(s >= 0)  # Ensure slack is non-negative

            # Big-M constraints (relax constraints when s[i] > 0)
            opti.subject_to(x_var[k] - xmax <= s[0] * M)  # Left of rectangle OR
            opti.subject_to(xmin - x_var[k] <= s[1] * M)  # Right of rectangle OR
            opti.subject_to(y_var[k] - ymax <= s[2] * M)  # Below rectangle OR
            opti.subject_to(ymin - y_var[k] <= s[3] * M)  # Above rectangle OR

            # Enforce at least one of these is active
            opti.subject_to(ca.sumsqr(s) >= 1)

    # Add initial conditions
    opti.subject_to(x_var[0] == x_init)
    opti.subject_to(y_var[0] == y_init)
    opti.subject_to(theta_var[0] == theta_init)

    # Add boundary conditions to follow reference path
    # for k in range(N+1):
        # opti.subject_to(x_var[k] == reference_x[k])
        # opti.subject_to(y_var[k] == reference_y[k])

    opti.subject_to(x_var[N] == x_final)
    opti.subject_to(y_var[N] == y_final)
    opti.subject_to(theta_var[N] == theta_final)

    # Solver options
    opti.solver('ipopt')

    # Solve the optimization problem
    sol = opti.solve()

    # Extract the optimized trajectory
    x_opt = sol.value(x_var)
    y_opt = sol.value(y_var)
    theta_opt = sol.value(theta_var)
    v_opt = sol.value(v_var)
    delta_opt = sol.value(delta_var)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_opt, y_opt, marker='o', label="Optimized Path")
    plt.plot(reference_x, reference_y, '--', marker='o', label="Reference Path")


    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.title('Optimized Trajectory vs. Reference Path')
    plt.grid()
    plt.show()

    xs = range(len(path))

    plt.plot(xs, x_opt, marker='o', label='x')
    plt.plot(xs, y_opt, marker='o', label='y')
    plt.plot(xs, theta_opt, marker='o', label='theta')
    plt.plot(xs, reference_x, marker='o', label='ref x')
    plt.plot(xs, reference_y, marker='o', label='ref y')
    plt.plot(xs, reference_theta, marker='o', label='ref theta')
    plt.legend()
    plt.show()
    
    return np.hstack((x_opt.reshape(-1, 1), y_opt.reshape(-1, 1), theta_opt.reshape(-1, 1)))

def optimize_trajectory(path, aabbs, N=20, dt=0.1, L=1.0, M=100):
    """
    Optimize trajectory for a nonholonomic robot given a kinematic path.

    Parameters:
        path: List of waypoints [(x, y, theta), ...] (initial guess)
        aabbs: List of obstacles [(xmin, xmax, ymin, ymax), ...]
        N: Number of time steps
        dt: Time step duration
        L: Wheelbase length
        M: Big-M constant for collision constraints

    Returns:
        Optimized trajectory (x, y, theta, v, delta)
    """
    opti = ca.Opti()

    # State Variables (x, y, theta)
    X = opti.variable(3, N)
    x, y, theta = X[0, :], X[1, :], X[2, :]

    # Control Variables (v, delta)
    U = opti.variable(2, N-1)
    v, delta = U[0, :], U[1, :]

    # Initial Guess
    for k in range(N):
        opti.set_initial(X[:, k], path[k].value)
    
    # Add initial conditions
    opti.subject_to(X[:, 0] == path[0].value)
    opti.subject_to(X[:, N-1] == path[-1].value)

    # Add boundary conditions to follow reference path
    # for k in range(N+1):
        # opti.subject_to(x_var[k] == reference_x[k])
        # opti.subject_to(y_var[k] == reference_y[k])

    # opti.subject_to(x_var[N] == x_final)
    # opti.subject_to(y_var[N] == y_final)
    # opti.subject_to(theta_var[N] == theta_final)

    # Kinematic Constraints (Discrete Integration)
    for k in range(N-1):
        x_next = x[k] + dt * v[k] * ca.sin(theta[k])
        y_next = y[k] + dt * v[k] * ca.cos(theta[k])
        theta_next = theta[k] + dt * delta[k]

        opti.subject_to(X[:, k+1] == ca.vertcat(x_next, y_next, theta_next))

    # Motion Constraints
    # opti.subject_to(v >= 0)  # No reverse motion
    opti.subject_to(v * delta == 0)  # Prevent turning while moving

    # Collision Avoidance (Big-M for AABBs)
    for (xmin, xmax, ymin, ymax) in aabbs:
        for k in range(N):
            s = opti.variable(4)  # Slack variables for OR conditions
            opti.subject_to(s >= 0)

            # Big-M constraints
            opti.subject_to(x[k] - xmax <= s[0] * M)  # Left of rectangle OR
            opti.subject_to(xmin - x[k] <= s[1] * M)  # Right of rectangle OR
            opti.subject_to(y[k] - ymax <= s[2] * M)  # Below rectangle OR
            opti.subject_to(ymin - y[k] <= s[3] * M)  # Above rectangle OR

            # At least one of the slack variables must be active
            opti.subject_to(ca.sumsqr(s) >= 1)

    # Objective: Minimize control effort
    opti.minimize(ca.sumsqr(v) + ca.sumsqr(delta))

    # Solver
    opti.solver('ipopt')

    # Solve the problem
    sol = opti.solve()

    # Extract optimized trajectory
    X_opt = sol.value(X)
    U_opt = sol.value(U)

    return X_opt, U_opt


if __name__ == '__main__':

    path = pickle.load(open('saved_paths/path.pickle', 'rb'))
    env = CarParkingEnv()
    aabbs = []
    for obs in env.obstacles:
        coordinates_array = np.asarray(obs.exterior.coords)[:-1]
        x_min = np.min(coordinates_array[:, 0])
        x_max = np.max(coordinates_array[:, 0])
        y_min = np.min(coordinates_array[:, 1])
        y_max = np.max(coordinates_array[:, 1])
        aabbs.append((x_min, x_max, y_min, y_max))

    # smoothed_path = traj_opt_smoothing(path)
    # print(smoothed_path)
    print(aabbs)
    print(path)
    X_opt, U_opt = optimize_trajectory(path, aabbs, N=len(path))
    print(X_opt.shape)
    # print(U_opt)
    X_opt = X_opt.T

    env.animate_path(X_opt, frame_delay=0.1)