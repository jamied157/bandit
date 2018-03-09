import numpy as np
import matplotlib.pyplot as plt


def normalbandit(means,plays,policy,policy_paramter=0.1):
	"""
	Simulates a stationary bandit with given means, plays and policy
	"""
	estimatemeans = np.zeros(len(means)) #initially these estiamtes are given higher values to encourage exploration of the space
	iterates = np.zeros(len(means)) #number of times each bandit has been selected
	value = np.zeros(plays + 1)
	pull_record = [[]]*(plays)
	for i in np.arange(plays):
		#in this loop we calculate the pull our policy does, record it, find it's reward, add the reward onto the total value
		#and update our estimates for the means then add to the count of the bandit pulled
		pull = policy(estimatemeans,policy_paramter)
		pull_record[i] = pull
		reward = np.random.normal(means[pull],1)
		value[i+1] = value[i] + reward
		estimatemeans[pull] = estimatemeans[pull] + (1/(1+iterates[pull]))*(reward - estimatemeans[pull])
		iterates[pull] += 1
	return [value,pull_record]

def non_stationary_bandit(bandit_no,plays,policy,policy_paramter = 0.1,discount1 = 0.9,discount2 = 0.95):
	"""
	Simulates a non_stationary bandit problem with means the same to begin with but then take a simple random walk
	Records number of optimal pulls and reward
	"""
	estimatemeans = np.zeros(bandit_no)
	means = np.zeros(bandit_no)
	iterates = np.zeros(bandit_no)
	value = np.zeros(plays + 1)
	pull_record = np.empty(plays)
	optimal_pull_record = np.zeros(plays)

	for i in np.arange(plays):
		#in this loop we calculate the pull our policy does, record it, find it's reward, add the reward onto the total value
		#and update our estimates for the means then add to the count of the bandit pulled the perform a simple random walk on the estimate means
		pull = policy(estimatemeans,policy_paramter)
		optimal_pull = np.argmax(means)
		if pull == optimal_pull:
			optimal_pull_record[i] = 1
		reward = np.random.normal(means[pull],1)
		value[i+1] = value[i] + reward
		estimatemeans[pull] = estimatemeans[pull] + (discount1)*(reward - estimatemeans[pull])
		estimatemeans = discount2*estimatemeans
		iterates[pull] += 1
		means = simple_random_walk(means)
	return [value,optimal_pull_record]


def epsilon_greedy(estimateMeans,epsilon):
	"""
	chooses a bandit to pull based on the epsilon greedy algorithm with epsilon chosen
	"""
	if np.random.uniform(0,1) > epsilon:
		pull = np.argmax(estimateMeans)
	else:
		pull = np.random.randint(len(estimateMeans))
	return pull

def softmax(estimateMeans,temp):
	"""
	Performs softmax action selection with temperature set by user
	"""
	a = np.array([0.]*(len(estimateMeans) +1))
	for i in np.arange(len(estimateMeans)):
		a[i+1] = a[i] + np.exp(estimateMeans[i]/temp)/sum(np.exp(estimateMeans/temp))
	r =  np.random.uniform(0,1)
	pull = np.argmax([prob for prob in a if prob < r])
	return pull

def simple_random_walk(a):
	"""
	takes an array and performs a simple random walk on each of the elements (i.e increases it with prob = 1/2, decreases with prob = 1/2)
	"""
	for i in np.arange(len(a)):
		if np.random.uniform(0,1) > 1/2:
			a[i] += 1
		else:
			a[i] += -1
	return a
