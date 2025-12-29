# Praval's Motion Planning Library

## Robots

### Point Robot

### Disc Robot

Parameters:
`Radius`

### Polygonal Robot
Parameters:
`Height`
`Width`

### Planar Mobile Arm
Parameters:
`Base Height?`
`Base Width?`
`Number of Links`
`Length of Links`


## Environments

### Deterministic Environments

#### Parking Space Environment

![Parking Space Environment](./assets/environments/parking_space.png)

### Probabilistic Environments

### Biased Passage
Parameters:
`Num Walls`
`Bias`

![Biased Passage Environment](./assets/environments/biased_passage.png)

### Random Sample Passage
Parameters:
`Num Walls`

![Random Sample Passage Environment](./assets/environments/random_sample_passage.png)

## Search Algorithms

This library mostly explores a class of motion planning algorithms known as sampling-based motion planning algorithms. This means each planning algorithm relies on probabilistic properties to alleviate some of the 

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