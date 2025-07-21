## General Description
The code is developed to evaluate the potential of using a deep reinforcement learning (RL) algorithm (i.e., Deep Q-Network) in automating the 2D stock-constrained layout design for reuse. 
It is part of a work for the journal article titled "Automating the Layout Design Process of 2D Construction Material Stocks under Uncertain Stock Geometry and Availability with Deep Reinforcement Learning", currently preparing for publication (7/21/2025).
More details can be found in the paper.

## Code Overview
### Folder structure
To run the proposed RL algorithm, reuse.py in the "envs" folder defines the RL environment.
The DQN model and learning process are encoded in files under "dqn".
The "utils" folder includes the files for rendering, input configuration, and the others.
The following Jupyter notebook files are developed to run the RL and GA.

### DQN Algorithm
run.ipynb is programmed to initiate the RL learning. Users can input the user-defined variables, run the RL algorithm, and analyze the results.

### Comparative approach: Genetic Algorithm
ga.ipynb can test the same process-based automated 2D reclaimed stock layout design approach using GA.

### Input format
Users can input variables under the jupyter notebook file.
The inputs include the environmental and computational variables.
1) "outer_shape" defines the width and height of the canvas (e.g., wall element).
2) "input_shape_n" is used to specify the [geometric variability, width, heigh, and count] of the stocks in the inventory. The paper describes the format in depth.
3) Computational variables, including the number of episodes, batch size, learning rate, and maximum steps allowed, can be specified.

## Contact Info
Seungah Suh, PhD student @ CEPM, UT Austin (sasuh@utexas.edu)
