# Model Predictive Control

# 

import numpy as np
import matplotlib.pyplot as plt
from utils import interpolate_edge
from sklearn.neighbors import KDTree
import casadi as ca

from space import DubinsCar

env = DubinsCar()

path = np.array([[0.0, 0.0],
                 [25.0, 0.0],
                 [25.0, 25.0],
                 [0.0, 25.0]])
full_path = np.vstack((path, path[0:1, :]))

interpolated_path = np.array([interpolate_edge(env.make_state(full_path[i]), env.make_state(full_path[i+1]), 0.5) for i in range(len(full_path)-1)])
interpolated_path = interpolated_path.reshape(-1, 2)
print(interpolated_path.shape)

kd_tree = KDTree(interpolated_path)
# dist, ind = kd_tree.query(current_state[:2])
# ind = ind.flatten()
# horizen_length = 50
# max_horizon_idx = horizon_length + ind # (WILL NEED TO BE NORMALIZED)


class DubinsCarSolver():
    def __init__(self, T=10, N=100):
        self.T = T 
        self.N = N
        self.dt = self.T / self.N

        self.L = 4

        self.opti = ca.Opti()

        # State: x, y, v, phi, theta

        self.xs = self.opti.variable(self.N+1)
        self.ys = self.opti.variable(self.N+1)
        self.vs = self.opti.variable(self.N+1)
        self.phis = self.opti.variable(self.N+1)
        self.thetas = self.opti.variable(self.N+1)

        self.accels = self.opti.variable(self.N)
        self.psis = self.opti.variable(self.N)

    def set_motion_constraints(self, xs, ys, vs, phis, thetas, accels, psis):
        for k in range(self.N):
            
            # X Transition
            self.opti.subject_to(xs[k+1] == xs[k] + self.dt * vs[k] * ca.cos(thetas[k]))

            # Y Transition
            self.opti.subject_to(ys[k+1] == ys[k] + self.dt * vs[k] * ca.sin(thetas[k]))

            # V Transition
            self.opti.subject_to(vs[k+1] == vs[k] + self.dt * accels[k])

            # Phi Transition
            self.opti.subject_to(phis[k+1] == phis[k] + self.dt * psis[k])

            # Theta Transition
            # v / self.car_length * np.tan(phi) * self.dt
            self.opti.subject_to(thetas[k+1] == thetas[k] + ((vs[k] / self.L) * ca.tan(phis[k]) * self.dt))

    def set_state_constraints(self):

        self.state_range = np.array([[-100, 100],
                                     [-100, 100],
                                     [-10, 10],
                                     [-np.pi/3, np.pi/3],
                                     [-np.inf, np.inf]])

        for k in range(1, self.N):
            # Set X Contraints
            self.opti.subject_to(self.xs[k] > self.state_range[0, 0])
            self.opti.subject_to(self.xs[k] < self.state_range[0, 1])

            # Set Y Contraints
            self.opti.subject_to(self.ys[k] > self.state_range[1, 0])
            self.opti.subject_to(self.ys[k] < self.state_range[1, 1])

            # Set V Contraints
            self.opti.subject_to(self.vs[k] > self.state_range[2, 0])
            self.opti.subject_to(self.vs[k] < self.state_range[2, 1])

            # Set Phi Contraints
            self.opti.subject_to(self.phis[k] > self.state_range[3, 0])
            self.opti.subject_to(self.phis[k] < self.state_range[3, 1])
            

    def set_control_constraints(self):
        self.control_range = np.array([[-5.0, 5.0],
                                       [-1.0, 1.0]])
        
        for k in range(self.N):
            # Set Accel Contraints
            self.opti.subject_to(self.accels[k] > self.control_range[0, 0])
            self.opti.subject_to(self.accels[k] < self.control_range[0, 1])

            # Set Accel Contraints
            self.opti.subject_to(self.psis[k] > self.control_range[1, 0])
            self.opti.subject_to(self.psis[k] < self.control_range[1, 1])

    def set_initial_conditions(self, starting_state):
        x, y, v, phi, theta = starting_state

        self.xs[0] = x
        self.ys[0] = y
        self.vs[0] = v
        self.phis[0] = phi
        self.thetas = theta

    def set_goal_conditions(self, goal_state, positional_only=True):
        x, y, v, phi, theta = goal_state

        self.xs[-1] = x
        self.ys[-1] = y

        if (not positional_only):
            self.vs[-1] = v
            self.phis[-1] = phi
            self.thetas[-1] = theta

    def solve(self):
        # Pick Solver
        self.opti.solver('ipopt')

        # Solve optimization problem
        solution = self.opti.solve()
        return solution

    def format_solution(self, solution):
        x_opt = solution.value(self.xs)
        y_opt = solution.value(self.ys)
        v_opt = solution.value(self.vs)
        phi_opt = solution.value(self.phis)
        theta_opt = solution.value(self.thetas)

        accel_opt = solution.value(self.accels)
        psi_opt = solution.value(self.psis)

        states = np.stack((x_opt, y_opt, v_opt, phi_opt, theta_opt), axis=1)
        controls = np.stack((accel_opt, psi_opt), axis=1)

        return states, controls
        
    def graph_solution(self, ax, solution):
        states, controls = self.format_solution(solution)
        ax.plot(states[:, 0], label='X')
        ax.plot(states[:, 1], label='Y')
        ax.plot(states[:, 2], label='V')
        ax.plot(states[:, 3], label='Phi')
        ax.plot(states[:, 4], label='Theta')

        ax.plot(controls[:, 0], label='Accel')
        ax.plot(controls[:, 1], label='Psi')

    # def 

def visualize_path(ax, path):
    full_path = np.vstack((path, path[0:1, :]))
    print(full_path)
    ax.plot(full_path[:, 0], full_path[:, 1], marker='o')


if __name__ == '__main__':
    
    # visualize_path(plt.gca(), path)
    visualize_path(plt.gca(), interpolated_path)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    env.draw_state(plt.gca(), state)
    plt.show()

