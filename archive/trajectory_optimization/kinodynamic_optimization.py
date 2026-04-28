import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from space import DubinsCar
# Car parameters

def trajectory_optimization(env, start, target):
    # Car parameters
    L = env.car_length  # Length of the car (meters)

    # Time horizon
    dt = env.dt
    N = 100  # number of discretization points

    # Define CasADi variables for state and control
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    v = ca.MX.sym('v')
    phi = ca.MX.sym('phi')
    theta = ca.MX.sym('theta')

    a = ca.MX.sym('a')  # acceleration
    psi = ca.MX.sym('psi')  # steering angle velocity

    # Kinematic model
    dx = v * ca.cos(theta)
    dy = v * ca.sin(theta)
    dtheta = v / L * ca.tan(psi)

    # Define state and control vectors
    states = ca.vertcat(x, y, v, phi, theta)
    controls = ca.vertcat(a, psi)

    # Create an empty list to store optimization variables
    state_traj = []
    control_traj = []

    # Set initial conditions and reference path (here we assume some predefined path)

    x_init, y_init, _, _, _ = start.value
    x_target, y_target, _, _, _ = target.value
    # v_init = 2.0  # Initial velocity
    # delta_init = 0.0  # Initial steering angle

    # Define the reference path (for example, a straight line with a slight turn)
    # reference_path = np.linspace(0, 10, N+1)
    # reference_x = reference_path
    # reference_y = np.sin(reference_path / 2)



    # Create optimization variables
    opti = ca.Opti()

    # Create decision variables
    x_var = opti.variable(N+1)
    y_var = opti.variable(N+1)
    v_var = opti.variable(N+1)
    phi_var = opti.variable(N+1)
    theta_var = opti.variable(N+1)
    
    a_var = opti.variable(N)
    psi_var = opti.variable(N)

    # Define cost function (minimize control effort)
    cost = 0
    for k in range(N):
        cost += a_var[k]**2 + psi_var[k]**2  # simple cost on control effort

    # Add constraints for kinematic dynamics
    for k in range(N):
        # dt = T / N  # Time step
        opti.subject_to(x_var[k+1] == x_var[k] + dt * v_var[k] * ca.cos(theta_var[k]-ca.pi/2))
        opti.subject_to(y_var[k+1] == y_var[k] + dt * v_var[k] * ca.sin(theta_var[k]-ca.pi/2))
        opti.subject_to(theta_var[k+1] == theta_var[k] + dt * v_var[k] / L * ca.tan(phi_var[k]))
        opti.subject_to(v_var[k+1] == v_var[k] + a_var[k] * dt)
        opti.subject_to(phi_var[k+1] == phi_var[k] + psi_var[k] * dt)

        opti.subject_to(psi_var[k] <= ca.pi/3)
        opti.subject_to(psi_var[k] >= -ca.pi/3)

    # Add initial conditions
    opti.subject_to(x_var[0] == x_init)
    opti.subject_to(y_var[0] == y_init)
    opti.subject_to(x_var[-1] == x_target)
    opti.subject_to(y_var[-1] == y_target)

    # Add control constraints 
    # opti.subject_to(psi <= ca.pi/3)
    # opti.subject_to(psi >= -ca.pi/3)

    # Add boundary conditions to follow reference path
    # for k in range(N+1):
        # opti.subject_to(x_var[k] == reference_x[k])
        # opti.subject_to(y_var[k] == reference_y[k])

    # opti.subject_to(x_var[100] == reference_x[100])
    # opti.subject_to(y_var[100] == reference_y[100])

    # Solver options
    opti.solver('ipopt')

    # Solve the optimization problem
    sol = opti.solve()

    # Extract the optimized trajectory
    x_opt = sol.value(x_var)
    y_opt = sol.value(y_var)
    theta_opt = sol.value(theta_var)
    phi_opt = sol.value(phi_var)
    v_opt = sol.value(v_var)
    
    # delta_opt = sol.value(delta_var)
    # print(len(x_opt), len(reference_x))
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

    a_opt = sol.value(a_var)
    psi_opt = sol.value(psi_var)

    # print(a_opt, psi_opt)

    print(x_opt, y_opt)

    controls = np.vstack((a_opt, psi_opt)).T
    return [(env.make_control(c), dt) for c in controls]

if __name__ == '__main__':
    # np.random.seed(0)
    env = DubinsCar()
    start, target = env.sample_task()

    control_seq = trajectory_optimization(env, start, target)
    state_seq = env.simulate(start, control_seq)
    env.animate_path(state_seq, frame_delay=0.001)

    print([s.value for s in state_seq])