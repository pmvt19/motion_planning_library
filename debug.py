from environments import DubinsCarEnv
import numpy as np 


if __name__ == '__main__':

    env = DubinsCarEnv()

    state = env.make_state(np.array([6.8738416, 7.59822891, 1.62531756, 0.44829068, 3.03860264]))
    control = env.make_control(np.array([2.76940055, 0.55987465]))
    print(state.value)
    print(env.extend_state(state, 0.4, control)[0].value)
    exit()