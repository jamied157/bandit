import bandit as bandit
import numpy as np

#Average value will keep track of the average value generated throughout the process while optimal play will keep track of the proportion
#of optimal plays made throughout it, each element represents a timestep in the process, each element is averaged over all 2000 processes.
average_value = np.array([0] * 1001)
optimal_play = np.array([0] * 1000)
for i in range(2000):
	[value,optimal_pull_record] = bandit.non_stationary_bandit(10,1000,bandit.epsilon_greedy,0.1)
	average_value = average_value + (1/(i+1))*(value - average_value)
	#have completed a bandit process anc updated the average value matrix
	optimal_play = optimal_play + (1/(i+1))*(optimal_pull_record - optimal_play)
	#have updated the optimal play matrix
	print(i)

average_reward = np.zeros(1001)
for i in np.arange(1000):
	average_reward[i+1] = average_value[i+1] - average_value[i]
