import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Car parameters
L = 2.5  # Length of the car (meters)

# Time horizon
T = 10  # seconds
N = 100  # number of discretization points


## ---- NOT USED ---- ##
# Define CasADi variables for state and control
x = ca.MX.sym('x')
y = ca.MX.sym('y')
theta = ca.MX.sym('theta')

v = ca.MX.sym('v')  # velocity
delta = ca.MX.sym('delta')  # steering angle

# Kinematic model
dx = v * ca.cos(theta)
dy = v * ca.sin(theta)
dtheta = v / L * ca.tan(delta)

# Define state and control vectors
states = ca.vertcat(x, y, theta)
controls = ca.vertcat(v, delta)

# Create an empty list to store optimization variables
state_traj = []
control_traj = []
## ---- NOT USED ---- ##

# Set initial conditions and reference path (here we assume some predefined path)
x_init = 0.0
y_init = 0.0
theta_init = 0.0
v_init = 2.0  # Initial velocity
delta_init = 0.0  # Initial steering angle

# Define the reference path (for example, a straight line with a slight turn)
reference_path = np.linspace(0, 10, N+1)
reference_x = reference_path
reference_y = np.sin(reference_path / 2)

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
# for k in range(N):
#     cost += v_var[k]**2 + delta_var[k]**2  # simple cost on control effort

# Add constraints for kinematic dynamics
for k in range(N):
    dt = T / N  # Time step
    opti.subject_to(x_var[k+1] == x_var[k] + dt * v_var[k] * ca.cos(theta_var[k]))
    opti.subject_to(y_var[k+1] == y_var[k] + dt * v_var[k] * ca.sin(theta_var[k]))
    # opti.subject_to(theta_var[k+1] == theta_var[k] + dt * v_var[k] / L * ca.tan(delta_var[k]))
    opti.subject_to(theta_var[k+1] == theta_var[k] + dt * delta_var[k])
    opti.subject_to(v_var[k] * delta_var[k] == 0)

    opti.subject_to(v_var[k] < 3)
    opti.subject_to(-3 < v_var[k])

# Add initial conditions
opti.subject_to(x_var[0] == x_init)
opti.subject_to(y_var[0] == y_init)
opti.subject_to(theta_var[0] == theta_init)

# Add boundary conditions to follow reference path
# for k in range(N+1):
    # opti.subject_to(x_var[k] == reference_x[k])
    # opti.subject_to(y_var[k] == reference_y[k])

opti.subject_to(x_var[100] == reference_x[100])
opti.subject_to(y_var[100] == reference_y[100])

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
print(len(x_opt), len(reference_x))
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

plt.plot(x_opt, label='x')
plt.plot(y_opt, label='y')
plt.plot(theta_opt, label='theta')
plt.plot(v_opt, label='v')
plt.plot(delta_opt, label='delta')
plt.legend()
plt.show()

print(np.stack((x_opt, y_opt, theta_opt), axis=1))

print(v_opt)
print(delta_opt)

from space import SkidSteerCar

env = SkidSteerCar()

controls = np.stack((v_opt, delta_opt), axis=1)
print(controls)
controls = [(c, 0.1) for c in controls]
state_seqs = env.simulate(env.make_state(np.array([0.0, 0.0, 0.0])), controls)
print([s.value for s in state_seqs])
env.animate_path(state_seqs)
