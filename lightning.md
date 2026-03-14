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

<p align="center">
<img src="./assets/search/lightning/database.png" alt="Path Database" width="75%">
</p>

2. Given a task in an environment, find N paths with the most similar tasks

<p align="center">
<img src="./assets/search/lightning/candidate_paths.png" alt="Candidate Paths" width="75%">
</p>

3. Validate the N paths, and find the path with the least amount of segments in collision

<p align="center">
<img src="./assets/search/lightning/selected_path.png" alt="Selected Path" width="75%">
</p>

4. Repair the segments in collision using RRT

<p align="center">
<img src="./assets/search/lightning/repairing_selected_path.png" alt="Repair Process with RRT" width="75%">
</p>

5. Return the final path (can smooth as well)

<p align="center">
<img src="./assets/search/lightning/final_path.png" alt="Final Path" width="75%">
</p>

