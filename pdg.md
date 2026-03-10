# Path Database Guidance

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

After generating a task $s,t$

## Offline Process:
```
Generate Path Database
```

## Online Process:
### Pseudocode:
```
for N iterations:
    ComputeRelevantPaths()

    connection_successful <- AttemptConnection()

    if connection_successful is True:
        collides_with_obst <- FollowPathUntilCollision()

        if collides_with_obst:
            DeletePathSegmentInCollision()
        else:
            return Path

    else:
        pdg.tree <- DoRRT(starting_tree=pdg.tree, num_steps=M)
```

# Search Step Figures:

## Retained Relevant Paths

<p align="center">
<img src="./assets/search/pdg/relevant_paths.png" alt="PDG Search" width="75%">
</p>

## Search Tree

<p align="center">
<img src="./assets/search/pdg/pdg.gif" alt="PDG Search" width="75%">
</p>

