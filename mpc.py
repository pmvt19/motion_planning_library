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

    def set_motion_constraints(self):
        for k in range(self.N):
            # print(self.thetas[k])
            # self.thetas[k] -= (ca.pi/2)
            # X Transition
            self.opti.subject_to(self.xs[k+1] == self.xs[k] + self.dt * self.vs[k] * ca.cos(self.thetas[k]))

            # Y Transition
            self.opti.subject_to(self.ys[k+1] == self.ys[k] + self.dt * self.vs[k] * ca.sin(self.thetas[k]))

            # V Transition
            self.opti.subject_to(self.vs[k+1] == self.vs[k] + self.dt * self.accels[k])

            # Phi Transition
            self.opti.subject_to(self.phis[k+1] == self.phis[k] + self.dt * self.psis[k])

            # Theta Transition
            # v / self.car_length * np.tan(phi) * self.dt
            self.opti.subject_to(self.thetas[k+1] == self.thetas[k] + ((self.vs[k] / self.L) * ca.tan(self.phis[k]) * self.dt))

    def set_state_constraints(self):

        self.state_range = np.array([[-100, 100],
                                     [-100, 100],
                                     [-3, 3],
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

        self.opti.subject_to(self.xs[0] == x)
        self.opti.subject_to(self.ys[0] == y)
        self.opti.subject_to(self.vs[0] == v)
        self.opti.subject_to(self.phis[0] == phi)
        self.opti.subject_to(self.thetas[0] == theta)

    def set_goal_conditions(self, goal_state, component_mask=[True, True, False, False, False]):
        x, y, v, phi, theta = goal_state

        if component_mask[0]:
            self.opti.subject_to(self.xs[self.N] == x)
        if component_mask[1]:
            self.opti.subject_to(self.ys[self.N] == y)
        if component_mask[2]:
            self.opti.subject_to(self.vs[self.N] == v)
        if component_mask[3]:
            self.opti.subject_to(self.phis[self.N] == phi)
        if component_mask[4]:
            self.opti.subject_to(self.thetas[self.N] == theta)
    
    def init_solver(self, starting_state, goal_state):
        
        
        self.set_initial_conditions(starting_state)
        self.set_goal_conditions(goal_state)
        self.set_motion_constraints()
        self.set_state_constraints()
        self.set_control_constraints()

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

def visualize_path(ax, path):
    full_path = np.vstack((path, path[0:1, :]))
    print(full_path)
    ax.plot(full_path[:, 0], full_path[:, 1], marker='o')


if __name__ == '__main__':
    
    # visualize_path(plt.gca(), path)

    # visualize_path(plt.gca(), interpolated_path)
    # state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # env.draw_state(plt.gca(), state)
    # plt.show()

    solver = DubinsCarSolver()
    starting_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    goal_state = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    solver.init_solver(starting_state=starting_state, goal_state=goal_state)
    solution = solver.solve()
    states, controls = solver.format_solution(solution)

    solver.graph_solution(plt.gca(), solution)
    plt.legend()
    plt.show()

    env = DubinsCar()
    s = env.make_state(starting_state)
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), s)
    plt.show()
    controls = [(c, 0.1) for c in controls]
    state_seq = env.simulate(starting_state=env.make_state(starting_state), control_seq=controls)
    for i in range(len(state_seq)-1):
        print("---")
        states[i, 4] = states[i, 4] % (2*np.pi)
        print(states[i])
        print(controls[i][0])
        print(state_seq[i].value)
        print("---")
    env.animate_path(state_seq)
    # print(np.array([s.value for s in state_seq]))
