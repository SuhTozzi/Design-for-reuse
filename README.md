## General Description
The code is developed to implement the propsoed framework for algorithm-based design optimization for reuse.
The code generates the optimal designs for reclaimed building materials using improved A* algorithm, heuristic algorithm, and genetic algorithm for pathfinding, stock assignment, and overall optimization structure, respectively.
It is part of a work for the journal article titled "Algorithm-based design optimization for reuse: stock assignment and path generation for reclaimed building materials", currently preparing for publication (4/23/2024).
More details can be found in the paper.

## Code Overview
### user_main.ipynb
Users can input variables and get the visualized results from this jupyter notebook file.
Users can define three basic geometric configuration-related variables and six optimization-related parameters.
Six optimization-related parameters follows the concept of the genetic algorithm: num_generations, num_population, num_parents_mating, random_prob, num_new_p, and weights.
Three variables for basic geometric configuration include targets, obstacles, and stocks. The unit of these variables depends on the users' decision.
1) "targets" requires XYZ coordinates of each target
2) "obstacles" is formmated as (lower top left corner x, y, z, w, h, d)
3) "stocks" refers to the length of each available stock. If there are three stocks available and their lengths are 3, 4, 4, the variable should be defined as "[3,4,4]".

### ga_opt.py
This python file is a collection of functions for a genetic algorithm-based path and stock assignment optimization

### pathfinding.py
This python file is developed to find a shortest paths and path lengths using the A* ("A star") algorithm.
NetworkX, a Python package for complex networks (https://networkx.org/), is used as a baseline.

### tools.py
This python file consists of supplementary codes for the other codes (e.g., reorganizing path after path generation, adding randomness to the optimization process).
Visualization is also located in this file.

## Contact Info
Seungah Suh (sasuh@utexas.edu)
