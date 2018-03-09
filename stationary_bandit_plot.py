import bandit as bd
import numpy as np
import matplotlib.pyplot as plt

#Average value will keep track of the average value generated throughout the process while optimal play will keep track of the proportion
#of optimal plays made throughout it, each element represents a timestep in the process, each element is averaged over all 2000 processes.
average_value = np.array([0] * 1001)
optimal_play = np.array([0] * 1000)
for i in range(2000):
	means = np.random.normal(0,1,10)
	optimal_action = np.argmax(means)
	#have generated the means and calculated the best action to take
	[value,pull_record] = bd.normalbandit(means,1000,bd.epsilon_greedy,0)
	average_value = average_value + (1/(i+1))*(value - average_value)
	#have completed a bandit process anc updated the average value matrix
	optimal_record = [1 if action == optimal_action else 0 for action in pull_record]
	optimal_play = optimal_play + (1/(i+1))*(optimal_record - optimal_play)
	#have updated the optimal play matrix

average_reward = np.zeros(1001)
for i in np.arange(1000):
	average_reward[i+1] = average_value[i+1] - average_value[i]
