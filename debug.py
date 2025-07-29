from environments import DubinsCarEnv
import numpy as np 
import torch
import time

if __name__ == '__main__':

    # env = DubinsCarEnv()

    # state = env.make_state(np.array([6.8738416, 7.59822891, 1.62531756, 0.44829068, 3.03860264]))
    # control = env.make_control(np.array([2.76940055, 0.55987465]))
    # print(state.value)
    # print(env.extend_state(state, 0.4, control)[0].value)
    # exit()

    states1 = np.random.random(size=(10000, 15))
    states2 = np.random.random(size=(10000, 15))

    start_time = time.time()
    sum_states = states1 + states2
    end_time = time.time()

    print(f"Time to sum: {end_time - start_time}")

    start_time = time.time()
    states1_torch = torch.tensor(states1)
    states2_torch = torch.tensor(states2)
    end_time = time.time()

    print(f"Time to convert from numpy array to torch tensor: {end_time - start_time}")

    start_time = time.time()
    sum_states_torch = states1_torch + states2_torch
    end_time = time.time()

    print(f"Time to sum torch: {end_time - start_time}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device('cuda')
    print(f"Using Device: {device}")

    start_time = time.time()
    states1_torch_gpu = states1_torch.to(device)
    states2_torch_gpu = states2_torch.to(device)
    end_time = time.time()

    print(f"Time to move data to gpu: {end_time - start_time}")

    start_time = time.time()
    sum_states_torch_gpu = states1_torch_gpu + states2_torch_gpu
    end_time = time.time()

    print(f"Time to sum torch gpu: {end_time - start_time}")






    
