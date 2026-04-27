import numpy as np

from motion_planning.tools import NumpyState
from motion_planning.space import RobotSpace


class NonHolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()

        self.dt = 0.1
    
    def make_control(self, state: np.ndarray):
        raise NotImplementedError
    def state_derivative(self, state, control):
        raise NotImplementedError
    def sample_controls(self):
        raise NotImplementedError
    def clip_state(self, state):
        raise NotImplementedError
    
    def simulate(self, starting_state: NumpyState, control_seq: list):
        state = starting_state
        state_seqs = [state]
        for control, time in control_seq:
            state, _, _ = self.extend_state(state, time, control, do_collision_checking=False)
            state_seqs.append(state)
        return state_seqs
    
    def extend_state(self, state: NumpyState, time: float, controls=None, do_collision_checking=True):
        if controls is None:
            controls = self.sample_controls()

        list_of_states = [state]
        running_time = 0
        num_iterations = int(time / self.dt)

        for i in range(num_iterations):
            state = self.simulate_step(state, controls)
            if do_collision_checking and not self.is_valid(state):
                break
            running_time = (i+1) * self.dt
            list_of_states.append(state)

        return list_of_states[-1], controls, running_time
    
    def simulate_step(self, state, control):
        state = self.get_state_value(state)
        x_dot = self.state_derivative(state, control)
        # Add Clipping of Values Here
        clipped_state = self.clip_state(state + x_dot)
        return self.make_state(clipped_state)
        # return self.make_state(state + x_dot)