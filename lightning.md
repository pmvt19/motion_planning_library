# Lightning

This file is intended to explain how PDG works:

# Environment Distribution

### Biased Passage

<p align="center">
<img src="./assets/search/pdg/environment.png" alt="Biased Passage Environment" width="75%">
</p>

# Database

<p align="center">
<img src="./assets/search/pdg/database.png" alt="PDG Database" width="75%">
</p>

# Algorithm Overview

1. Generate a path database
2. Given a task in an environment, find N paths with the most similar tasks
3. Validate the N paths, and find the path with the least amount of segments in collision
4. Repair the segments in collision using RRT
5. Return the final path (can smooth as well)