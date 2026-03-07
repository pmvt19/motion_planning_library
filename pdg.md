# Path Database Guidance

This file is intended to explain how PDG works:

# Environment Distribution

### Biased Passage

<p align="center">
<img src="./assets/pdg/biased_passage.png" alt="Biased Passage Environment" width="75%">
</p>

# Database

<p align="center">
<img src="./assets/pdg/database.png" alt="PDG Database" width="75%">
</p>

# Algorithm Overview

After generating a task $s,t$

## Step 1: Compute Relevant Paths

<!-- <p align="center">
<img src="./assets/pdg/biased_passage.png" alt="Biased Passage Environment" width="75%">
</p> -->

* Image with database paths only leading to the target

## Step 2: Attempt Connection to Paths

* Image with remaining paths and search tree attempting connections

## Step 3a: Follow Connection Path Until Collision

* Image with path that would lead to a collision eventually

## Step 3c: Delete Collision Portion

* Image with tree to path that would be deleted

## Step 3b: If No Connection Attempts are Successful, do RRT

## Step 4: Repeat
Repeat 