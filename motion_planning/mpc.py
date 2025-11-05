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

interpolated_path = np.array([interpolate_edge(env.make_state(full_path[i]), env.make_state(full_path[i+1]), 0.15) for i in range(len(full_path)-1)])
interpolated_path = interpolated_path.reshape(-1, 2)
print(interpolated_path.shape)

kd_tree = KDTree(interpolated_path)
# dist, ind = kd_tree.query(current_state[:2])
# ind = ind.flatten()
# horizen_length = 50
# max_horizon_idx = horizon_length + ind # (WILL NEED TO BE NORMALIZED)


class DubinsCarSolver():
    def __init__(self, T=5, N=50):
        self.T = T 
        self.N = N
        # self.dt = self.T / self.N
        self.dt = 0.1

        self.L = 4

        self.opti = ca.Opti()

        self.set_optimization_variables()
    
    def set_optimization_variables(self):
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
        self.control_range = np.array([[-1.0, 5.0],
                                       [-1.0, 1.0]])
        
        for k in range(self.N):
            # Set Accel Contraints
            self.opti.subject_to(self.accels[k] > self.control_range[0, 0])
            self.opti.subject_to(self.accels[k] < self.control_range[0, 1])

            # Set Accel Contraints
            self.opti.subject_to(self.psis[k] > self.control_range[1, 0])
            self.opti.subject_to(self.psis[k] < self.control_range[1, 1])

    def set_initial_conditions(self, start_state):
        x, y, v, phi, theta = start_state

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

    def set_path_cost(self, path):
        # assert(len(path) == self.N)
        cost_function = 0
        # INFO: Path may or may not include the starting state
        # TODO: Deal with this case
        # for i in range(1, len(path)):
        for i in range(len(path)-1, len(path)):
            cost_function += ((path[i, 0] - self.xs[i])**2 + (path[i, 1] - self.ys[i])**2)

        for i in range(1, len(path)//2):
            cost_function += ((path[-1, 0] - self.xs[i])**2 + (path[-1, 1] - self.ys[i])**2)
        self.opti.minimize(cost_function)
    
    def init_solver(self, start_state, goal_state, path):
        self.set_initial_conditions(start_state)
        # self.set_goal_conditions(goal_state)
        self.set_motion_constraints()
        self.set_state_constraints()
        self.set_control_constraints()
        self.set_path_cost(path)

    def solve(self):

        # Define solver options to suppress printing
        opts = {
            'ipopt.print_level': 0,  # Main verbosity level
            'print_time': 0,         # Do not print information about execution time
            'ipopt.sb': 'yes'        # Suppress the IPOPT license banner
        }

        # Pick Solver
        self.opti.solver('ipopt', opts)

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
    ax.plot(full_path[:, 0], full_path[:, 1], marker='o')

class MPC():
    def __init__(self, env=DubinsCar(), solver_type=DubinsCarSolver):
        self.env = env
        self.solver_type = solver_type

        # self.horizon_dist = 50
        self.horizon_dist = 25
        # self.path = None

        ## --- TEMP -- ##
        # Move out of here
        # path = np.array([[0.0, 0.0],
        #          [25.0, 0.0],
        #          [25.0, 25.0],
        #          [0.0, 25.0]])

        # path = np.array([[0.0, 0.0],
        #          [7.0, 0.0],
        #          [7.0, 7.0],
        #         # [10.0, 10.0],
        #          [0.0, 7.0]])
        

        path = np.array([[0.0, 0.0],
                 [5.0, 5.0],
                 [10.0, 4.0],
                # [10.0, 10.0],
                 [14.0, 10.0]])
        path = path * 2
        # full_path = np.vstack((path, path[0:1, :]))
        full_path = path

        # interpolated_path = np.array([interpolate_edge(env.make_state(full_path[i]), env.make_state(full_path[i+1]), 0.15) for i in range(len(full_path)-1)])
        print([interpolate_edge(env.make_state(full_path[i]), env.make_state(full_path[i+1]), 0.1).shape for i in range(len(full_path)-1)])
        interpolated_path = [interpolate_edge(env.make_state(full_path[i]), env.make_state(full_path[i+1]), 0.25) for i in range(len(full_path)-1)]
        # interpolated_path = interpolated_path.reshape(-1, 2)
        interpolated_path = np.vstack((interpolated_path))
        print(interpolated_path.shape)
        # exit()
        print(interpolated_path, len(interpolated_path))
        ## --- TEMP -- ##

        self.path = interpolated_path

    def mpc_step(self, state, path_segment):
        # TODO: Uncomment
        # assert(state is part of path_segment)

        # ASSUME: State is a part of path segment

        self.solver = self.solver_type(N=self.horizon_dist)
        goal_loc = path_segment[-1]
        fake_goal_state = np.array([goal_loc[0], goal_loc[1], 0.0, 0.0, 0.0])
        self.solver.init_solver(start_state=state, goal_state=fake_goal_state, path=path_segment)
        solution = self.solver.solve()
        states, controls = self.solver.format_solution(solution)

        control = controls[0]
        print(np.round(state, 2), np.round(control, 2))
        new_state = self.env.simulate_step(state, control)
        # new_state = states[1]

        return state, new_state, states, control
    
    def locate_on_path(self, state):
        kd_tree = KDTree(self.path)
        dist, ind = kd_tree.query([state[:2]])
        ind = ind.flatten()[0]
        print(f"Choosing Ind: {ind}")
        return ind+1
        
    def get_path_horizon(self, cur_idx):
        horizon_idx = cur_idx + self.horizon_dist # Normalize
        horizon_idx = min(horizon_idx, len(self.path))
        return horizon_idx
    
    def get_path_segment(self, start_idx, end_idx):
        path_segment = self.path[start_idx:end_idx]
        return path_segment

    def visualize(self, ax, state, path_horizon):
        self.env.draw_environment(ax)
        self.env.draw_state(ax, state)
        ax.plot(path_horizon[:, 0], path_horizon[:, 1], color='green', marker='o', zorder=2)

        full_path = np.vstack((self.path, self.path[0:1, :]))
        ax.plot(full_path[:, 0], full_path[:, 1], color='blue', marker='o', zorder=0)
    
    def run(self, start_state, visualize=False):
        state = start_state
        for i in range(1000):
            idx = self.locate_on_path(state)
            horizon_idx = self.get_path_horizon(idx)
            path_segment = self.get_path_segment(idx, horizon_idx)

        
            state, new_state, states, control = self.mpc_step(state, path_segment)
            state = new_state
            state = self.env.get_state_value(state)
            if visualize:
                # state or new_state here?
                # I think state...???
                plt.cla()
                self.visualize(plt.gca(), state, states)
                # plt.show()
                plt.pause(0.1)
        
        




if __name__ == '__main__':
    
    # visualize_path(plt.gca(), path)

    # visualize_path(plt.gca(), interpolated_path)
    # state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # env.draw_state(plt.gca(), state)
    # plt.show()

    path_segment = interpolated_path[:51, :]
    print(path_segment.shape)
    # exit()

    solver = DubinsCarSolver()
    start_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # start_state = np.array([6.63, 0.27, 0.06, 1.01, 0.56])
    goal_state = np.array([path_segment[-1, 0], path_segment[-1, 1], 0.0, 0.0, 0.0])
    # print(start_state)
    # print(goal_state)
    # print(path_segment)
    # exit()
    solver.init_solver(start_state=start_state, goal_state=goal_state, path=path_segment)
    solution = solver.solve()
    states, controls = solver.format_solution(solution)

    solver.graph_solution(plt.gca(), solution)
    plt.legend()
    plt.show()

    env = DubinsCar()
    env.x_range = [-5, 30]
    env.y_range = [-5, 30]
    s = env.make_state(start_state)
    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), s)
    visualize_path(plt.gca(), interpolated_path)
    # plt.plot(path_segment[:, 0], path_segment[:, 1], label='path', marker='o')
    plt.plot(states[:, 0], states[:, 1], color='green', label='mpc path', marker='o')
    
    controls = [(c, 0.1) for c in controls]
    state_seq = env.simulate(starting_state=env.make_state(start_state), control_seq=controls)
    state_seq_np = np.array([s.value for s in state_seq])
    plt.plot(state_seq_np[:, 0], state_seq_np[:, 1], color='red', label='simulated path', marker='o')

    plt.legend()
    plt.show()
    for i in range(len(state_seq)-1):
        print("---")
        states[i, 4] = states[i, 4] % (2*np.pi)
        print(states[i])
        print(controls[i][0])
        print(state_seq[i].value)
        print("---")
    # env.animate_path(state_seq)
    # print(np.array([s.value for s in state_seq]))
    
    mpc = MPC(env=env)
    mpc.run(start_state, visualize=True)
