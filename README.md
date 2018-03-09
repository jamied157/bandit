# Bandit
### This project was done alongside reading through the Sutton and Barto book Reinforcement Learning, it is based on the first chapter on Bandit processes.

The module contains the code for two bandit different processes: one which is a stationary process where the task of the algorithm is to learn the best bandit lever to pull, the other is non-stationary and requires the algorithm to discount information from earlier pulls to allow for the fact that a lever may change in how favourable the rewards it produces are. Two algorithms are put in the module also - the softmax and epsilon greedy algorithms.

The two other files contain scripts to allow the easy plotting of the results of an average performance of each of these algorithms over many iterations.
