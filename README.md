# Praval's Motion Planning Library

## Environments

### Deterministic Environments

#### Parking Space Environment

<!-- ![Parking Space Environment](./assets/environments/parking_space.png) -->
<p align="center">
<img src="./assets/environments/parking_space.png" alt="Parking Space Environment" width="75%">
</p>

### Probabilistic Environments

### Biased Passage
Parameters:
`Num Walls`
`Bias`

<!-- ![Biased Passage Environment](./assets/environments/biased_passage.png) -->
<p align="center">
<img src="./assets/environments/biased_passage.png" alt="Biased Passage Environment" width="75%">
</p>

### Random Sample Passage
Parameters:
`Num Walls`
`Gap Width`

<!-- ![Random Sample Passage Environment](./assets/environments/random_sample_passage.png) -->
<p align="center">
<img src="./assets/environments/random_sample_passage.png" alt="Random Sample Passage Environment" width="75%">
</p>

## Robots

### Point Robot

### Disc Robot

Parameters:
`Radius`

<p align="center">
<img src="./assets/robots/disc_robot.gif" alt="Disc Robot" width="75%">
</p>

### Polygonal Robot
Parameters:
`Height`
`Width`

< PLACEHOLDER >

<!-- ![Random Sample Passage Environment](./assets/robots/polygonal_robot.gif) -->
<p align="center">
<img src="./assets/robots/polygonal_robot.gif" alt="Polygonal Robot" width="75%">
</p>

### Planar Mobile Arm
Parameters:
`Base Height?`
`Base Width?`
`Number of Links`
`Length of Links`

<p align="center">
<img src="./assets/robots/planar_mobile_arm.gif" alt="Planar Mobile Arm Robot" width="75%">
</p>


## Search Algorithms

This library mostly explores a class of motion planning algorithms known as sampling-based motion planning algorithms. This means each planning algorithm relies on probabilistic properties to alleviate some of the heavy compuation reqired for classical motion planning algorithms.

### RRT

< PLACEHOLDER >

<p align="center">
<img src="./assets/rrt.gif" alt="RRT GIF" width="75%">
</p>

#### Why this works

RRT is inherently biased to expore open regions of the $\mathcal{C}$-Space first, then fill in the gaps later due to the **voronoi bias**. 

<!-- ![RRT Voronoi Diagram](./assets/rrt_voronoi_diagram.png) -->
<p align="center">
<img src="./assets/rrt_voronoi_diagram.png" alt="RRT Voronoi Diagram" width="75%">
</p>

### Bi-Directional RRT

Bi-Directional RRT attempts to grow two trees: one from the start (just like RRT) and one from the target. If the trees have any nodes within a certain thresholded distance between them and the edge connecting the two nodes are valid, then the trees connect and a valid path between the start and the target is found.

The following is a GIF showing the start tree (orange) and the target tree (blue) growing towards each other and connecting in the middle:
<p align="center">
<img src="./assets/bidir_rrt.gif" alt="BiDirectional RRT GIF" width="75%">
</p>

### RRT*

Unlike RRT, RRT* attempts to find an asymptotically optimal path (i.e. the shortest path between configuration $p$ and $q$). This is done by adding an additional step to the RRT expansion step: rewiring. The goal of the rewiring step is to make the newly added node in the tree have the shortest path to the starting (root) configuration using the existing tree structure. 


This means that the tree will not stop building after finding an initial path from $p$ to the $q$; it will continuosly search, refining the path until it hits its maximum runtime.

### RSG

### PRM (Probabiilistic Roadmap)

Unlike RRT which is a single query planner, PRM generates a roadmap that can be required for multiple tasks within the same configuration space. The bulk of the computation for PRM is done during the creation of the roadmap. 


The standard PRM algorithm is as follows:

1. Sample N Random Points in the Configuration Space
2. Validate All Sampled Points and Remove Invalid Points
3. Attempt Edge Connections with either M Neighbors or All Other Points within Radius R and Keep Only Valid Edges
4. Attach the `start` and `target` nodes to the roadmap and validate their edges
5. Use a shortest path algorithm such as Dijkstra's or A* to solve for the path

The following is a GIF of the generation of a PRM
<p align="center">
<img src="./assets/prm.gif" alt="PRM Generation" width="75%">
</p>

### Incremental PRM

Similar to traditional PRM, Incremental PRM also builds a roadmap in the $\mathcal{C}$-Space, but will continue to add nodes until the start and target configuration are in the same connected component. Once $p$ and $q$ are in the same connected component, this means there exists a valid path from start to the target, and thus this search algorithm can stop extending the graph and return the shortest path connecting the start and target via the existing roadmap.

The following is a GIF of Incremental PRM iteratively extending the roadmap to connect the start and target:
<p align="center">
<img src="./assets/incremental_prm.gif" alt="Incremental PRM Generation" width="75%">
</p>

### NonUniform PRM

Typically, building PRMs sample uniformly in the robot's $\mathcal{C}$-Space; however, many nodes in open areas of the $\mathcal{C}$-Space may not be as beneficial as nodes on the border of $\mathcal{C}_{free}$ and $\mathcal{C}_{obst}$.

NonUniform PRM works by initially sampling a set of nodes in the configuration space uniformly (just like traditional PRM), adding some normally distributed noise to the points, and comparing if one of the original points or noise affected points are in $\mathcal{C}_{free}$ while the other is in $\mathcal{C}_{obst}$. This means that we retain the node only if exactly one of the configuration (either the originally sampled one or the noise affected one) is in $\mathcal{C}_{free}$. This ensures that retained nodes are typically near $\mathcal{C}_{obst}$ regions.

The following is an example of a roadmap generated via the NonUniform PRM algorithm:
<p align="center">
<img src="./assets/nonuniform_prm.png" alt="Non-Uniform PRM" width="75%">
</p>

### Lazy PRM

Lazy PRM works almost exactly like PRM, but delays the most computationally expensive part of the algorithm: the edge collision checks. In traditional PRM, we validate all the edges of our roadmap before attempting to find a path from the start to the target. 

In contrast, Lazy PRM does not validate edges before attempting to search for a path. Instead Lazy PRM validates the edges of only potential paths connecting the start to the target. 

Lazy PRM searchs for an initial potential path in the roadmap. If found, it then validates all the edges in the potential path. If there are no invalid edges in the potential path, the algorithm will simply return that path as the final path. However, if there are invalid edges in the path, Lazy PRM will remove those edges from the roadmap and search again from a path from the start to the target. This will repeat until either a path is found with no invalid edges or it reaches a maximum number of iterations.

The following is a GIF of Lazy PRM Searching for a path with the invalid edges marked in pink:
<p align="center">
<img src="./assets/lazy_prm.gif" alt="Lazy PRM" width="75%">
</p>

# Visualizations

## 2D C-Space

< PLACEHOLDER >

Disc Robot Workspace and $\mathcal{C}$-Space Visualization

![2D C-Space Disc Robot](./assets/disc_robot_cspace_viz.gif)

Fixed Arm Workspace and $\mathcal{C}$-Space Visualization

![2D C-Space Robot Arm](./assets/Robot_Arm.gif)

<!-- ## 3D C-Space
![3D C-Space Polygonal Robot](./assets/Polygonal_Robot.png) -->

# Accelerated Collision Checks

## Circle Approximations For the Environment and Robot

# Acknowledgements
This repository is heavily inspired by work done during my time at the Parasol Lab working with, at the time, PhD candidate Amnon Attali. A few items in this codebase are a reimplementation intended to better my skills with NumPy operations. 