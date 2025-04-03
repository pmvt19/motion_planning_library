import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict

x_range = [-10,10]
y_range = [-10,10]

def sample_single_point(range):
    x = (np.random.random() * (range[1] - range[0])) + range[0]
    return x 

def sample_point(x_range, y_range):
    x = sample_single_point(x_range)
    y = sample_single_point(y_range)
    return (x, y)

def select_node(tree, target, goal_bias=0):
    if np.random.random() < goal_bias:
        x, y = target
    else:
        x, y = sample_point(x_range, y_range)
    nodes = np.array([node for node in tree.keys()])
    # print(len(nodes))
    kdt = KDTree(nodes)
    dist, ind = kdt.query(np.array([[x,y]]), k=1)
    idx = ind[0][0]
    return tuple(nodes[idx]), (x, y)

def expand_node(tree, node, sampled_point, target, delta=0.5):

    if np.linalg.norm((np.array(target) - np.array(node))) < delta:
        new_node = target
    else:
        dir = (np.array(sampled_point) - np.array(node)) / np.linalg.norm(np.array(sampled_point) - np.array(node))
        ext_amount = np.random.random() * delta
        new_node = node + (ext_amount * dir)
        new_node = tuple([c for c in new_node])
    
    tree[node].append(new_node)
    tree[new_node]
    return new_node

def draw_tree(tree, start, target, hold=False):
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    nodes = np.array([node for node in tree.keys()])
    plt.scatter(nodes[:, 0], nodes[:, 1])
    plt.scatter(start[0], start[1], s=100, c='green')
    plt.scatter(target[0], target[1], s=100, c='red')

    for p in tree:
        for c in tree[p]:
            plt.plot([p[0], c[0]], [p[1], c[1]])
    if hold:
        plt.show()
    plt.pause(0.01)
    plt.clf()


start = (0, 0)
target = (9, 9)
tree = defaultdict(list)
tree[start] = []
cur_node = start
num_steps = 0
max_steps = 2000
while (cur_node != target and num_steps < max_steps):
    print(f"Searching Step: {num_steps}")
    exp_node, sampled_point = select_node(tree, target, goal_bias=0)
    cur_node = expand_node(tree, exp_node, sampled_point, target, delta=1)
    num_steps += 1
    # print(cur_node)
    draw_tree(tree, start, target)

draw_tree(tree, start, target, hold=True)