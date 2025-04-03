from rrt import search
import matplotlib.pyplot as plt
import numpy as np 

goal_biases = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.9, 0.95]
times = []
num_runs = 5

for goal_bias in goal_biases:
    ablation_res = []
    for i in range(num_runs):
        search_time, _ = search((0,0), (9,9), goal_bias=goal_bias)
        ablation_res.append(search_time)
    times.append(np.mean(ablation_res))


plt.plot(goal_biases, times)
plt.show()

