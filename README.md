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

### Polygonal Robot
Parameters:
`Height`
`Width`

< PLACEHOLDER >

<!-- ![Random Sample Passage Environment](./assets/robots/polygonal_robot.gif) -->
<p align="center">
<img src="./assets/robots/polygonal_robot.gif" alt="Parking Space Environment" width="75%">
</p>

### Planar Mobile Arm
Parameters:
`Base Height?`
`Base Width?`
`Number of Links`
`Length of Links`


## Search Algorithms

This library mostly explores a class of motion planning algorithms known as sampling-based motion planning algorithms. This means each planning algorithm relies on probabilistic properties to alleviate some of the heavy compuation reqired for classical motion planning algorithms.

### RRT

< PLACEHOLDER >

![RRT GIF](./assets/RRT_GIF.gif)

#### Why this works

![RRT Voronoi Diagram](./assets/rrt_voronoi_diagram.png)

### Bi-Directional RRT

### RRT*

### RSG

### PRM

< PLACEHOLDER >

![PRM Generation](./assets/prm.gif)

### Incremental PRM

### NonUniform PRM

### Lazy PRM

# Visualizations

## 2D C-Space

< PLACEHOLDER >

![2D C-Space Disc Robot](./assets/Disc_Robot_Motion.gif)

< PLACEHOLDER >

![2D C-Space Robot Arm](./assets/Robot_Arm.gif)

## 3D C-Space

![3D C-Space Polygonal Robot](./assets/Polygonal_Robot.png)

# Acknowledgements
This repository is heavily inspired by work done during my time at the Parasol Lab working with, at the time, PhD candidate Amnon Attali. A few items in this codebase are a reimplementation intended to better my skills with NumPy operations. 