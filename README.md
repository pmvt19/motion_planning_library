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

![Biased Passage Environment](./assets/environments/biased_passage.png)

### Random Sample Passage
Parameters:
`Num Walls`
`Gap Width`

![Random Sample Passage Environment](./assets/environments/random_sample_passage.png)

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

<!-- ![RRT Voronoi Diagram](./assets/rrt_voronoi_diagram.png) -->
<p align="center">
<img src="./assets/rrt_voronoi_diagram.png" alt="RRT Voronoi Diagram" width="75%">
</p>

### Bi-Directional RRT

### RRT*

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
![PRM Generation](./assets/prm.gif)

### Incremental PRM

### NonUniform PRM

### Lazy PRM

# Visualizations

## 2D C-Space

< PLACEHOLDER >

![2D C-Space Disc Robot](./assets/disc_robot_cspace_viz.gif)

< PLACEHOLDER >

![2D C-Space Robot Arm](./assets/Robot_Arm.gif)

## 3D C-Space

![3D C-Space Polygonal Robot](./assets/Polygonal_Robot.png)

# Acknowledgements
This repository is heavily inspired by work done during my time at the Parasol Lab working with, at the time, PhD candidate Amnon Attali. A few items in this codebase are a reimplementation intended to better my skills with NumPy operations. 