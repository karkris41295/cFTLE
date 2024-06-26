# How to use these files


Paper available here: https://arxiv.org/abs/2304.03326

In order to run the python scripts, it's necessary to have MPCTools available here: https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/

Another special package used in this work is the multiprocessing package for using multiple CPU cores for faster calculations.

This repository has two python scripts.

1) cFTLE_V.py - Code that visualizes the value function along with the cFTLE ridges (there is an option to load data, so some cells can be skipped when running cell by cell)
2) rl_ex2.py - Code that runs the reinforcement learning example and visualizes the cFTLE ridges. (This uses PyTorch and Stable Baselines packages!)

Some lines in the code call ftle_th6.npz/ftle_th15.npz, these are passive FTLE data which can be generated using code from single-agent-MPC-FTLE.

Recommendation is to run the code cell by cell in an IDE like spyder. Several lines commented out can be uncommented (Mostly related to plotting).
