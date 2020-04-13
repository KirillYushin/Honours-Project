import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
import csv
from game import Game
from nn import NeuralNetwork
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'nextState'])

class ExperienceReplay:
	def __init__(self, capacity):
		self.buffer = collections.deque(maxlen=capacity)

	def append(self, experience):
		self.buffer.append(experience)

	def sample(self, batchSize):
		indices = np.random.choice(len(self.buffer), batchSize, replace=False)
		states, actions, rewards, nextStates = zip(*[self.buffer[i] for i in indices])
		return np.array(states), np.array(actions), np.array(rewards), np.array(nextStates)


game = Game("train", True)
replayBuffer = 1000
replayMemory = ExperienceReplay(replayBuffer)
gameStateSize = game.car.radar.numBeams + 1
numActions = 6
epsilon = 1.0                # probability of doing a random move (exploration vs exploitation)
epsilonEnd = 0.05
gamma = 0.95                 # discount factor for future reward
batchSize = 32               # how many experiences to use for each training step
trainFreq = 10               # how often to perform a training step
updateFreq = trainFreq * 20  # how often to update weights of targetQN
mainQN = NeuralNetwork(gameStateSize, numActions, "main")
targetQN = NeuralNetwork(gameStateSize, numActions, "target")
lossPerEpoch = collections.deque()
accumulatedLoss = 0
averageQvalueEstimates = collections.deque()
QvalueAccum = 0
estimateCounter = 0

saver = tf.train.Saver()
with tf.Session() as sess:
	#'''
	#saver.restore(sess, "./checkpoints/session_duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_(e=0.02).ckpt")
	sess.run(tf.global_variables_initializer())
	
	frame = 0
	state = game.get_state()                                        # initial state
	while frame <= 150000:
		if (np.random.rand() < epsilon):                            # Select random action with probability epsilon (exploration)
			action = np.random.randint(0, numActions)
		else:                                                       # select action with the highest Q-value (exploitation)
			action = sess.run(mainQN.predict_op, feed_dict={mainQN.inputs: np.atleast_2d(state)})[0]
		
		reward = game.perform_action(action)                 # Perform selected action in the game and get the reward
		newState = game.get_state()                                 # updated state after the action
		if (game.quit):
			break
		
		# Save experience
		exp = Experience(state, action, reward, newState)
		replayMemory.append(exp)
		state = newState
		
		# Training step once every trainFreq frames (experience replay)
		if (frame > replayBuffer and (frame % trainFreq == 0)):
			states, actions, rewards, nextStates = replayMemory.sample(batchSize)
			Qvalues = sess.run(mainQN.predictions, feed_dict={mainQN.inputs: states})                   # predicted Q values for current state (for each experience)
			bestFutureActions = sess.run(mainQN.predict_op, feed_dict={mainQN.inputs: nextStates})      # best action agent can do in the next state (for each experience)
			targetQvalues = sess.run(targetQN.predictions, feed_dict={targetQN.inputs: nextStates})     # Q values for the next state from the target Q network (for each experience)
			#targetQvalues = sess.run(mainQN.predictions, feed_dict={mainQN.inputs: nextStates})
			
			# Calculate target Q values for each mini-batch experience
			for i in range(0, batchSize):
				if (rewards[i] < 0):
					Qvalues[i][actions[i]] = rewards[i]
				else:
					# Bellman equation for the Q function
					# (Double DQN) The action with the highest Q value from the next state is selected from the main network,
					# but the Q value (future reward) of that action is taken from the target network
					Qvalues[i][actions[i]] = rewards[i] + (gamma * targetQvalues[i][bestFutureActions[i]])
					QvalueAccum += Qvalues[i][actions[i]]
					estimateCounter += 1
			
			# Record loss
			accumulatedLoss += sess.run(mainQN.loss, feed_dict={mainQN.inputs: states, mainQN.targets: Qvalues})
			
			# Train on the mini-batch of experiences
			sess.run(mainQN.train_op, feed_dict={mainQN.inputs: states, mainQN.targets: Qvalues})

		# Update target Q network weights once every updateFreq frames (Double DQN)
		if (frame > replayBuffer and (frame % updateFreq == 0)):
			targetQN.copy_weights(sess, mainQN)
			lossPerEpoch.append(accumulatedLoss)
			accumulatedLoss = 0
			averageQvalueEstimates.append(round(QvalueAccum / estimateCounter, 2))
			QvalueAccum = 0
			estimateCounter = 0
		
		# Decrement epsilon over time
		if (epsilon > epsilonEnd):
			epsilon -= 0.00002
			
		if (epsilon > epsilonEnd and (frame % 5000 == 0)):
			print("Epsilon:", round(epsilon, 2))
		
		# Save session
		if (frame % 10000 == 0):
			saver.save(sess, "./checkpoints/session_duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_(e=0.02)_+20k.ckpt")
			print("Frame:", frame)
			
		frame += 1
	
	# Save crashes per lap
	with open('./stats/crashesPerLap.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(game.racetrack.crashesPerLap)
		
	# Save distance travelled until crash
	game.car.crash()
	with open('./stats/distanceTravelledUntilCrash.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(game.car.distanceTravelledUntilCrash)
		
	# Save mini-batch loss
	with open('./stats/lossPerEpoch.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(lossPerEpoch)
		
	# Save average Q value Estimates per epoch
	with open('./stats/averageQvalueEstimates.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(averageQvalueEstimates)
	
	# Run statistics
	#label = "Adam optimiser"
	averageDistanceTravelled = []
	i = 0
	accumDist = 0
	try:
		while True:
			accumDist += game.car.distanceTravelledUntilCrash.popleft()
			i += 1
			if (i == 10):
				averageDistanceTravelled.append(round(accumDist / 10, 2))
				i = 0
				accumDist = 0

	except IndexError:
		if (i != 0):
			averageDistanceTravelled.append(round(accumDist / i, 2))
	
	averageDistanceTravelled = np.array(averageDistanceTravelled)
	crashes = np.arange(1, len(averageDistanceTravelled) + 1, 1)
	plt.figure(1)
	plt.plot(crashes, averageDistanceTravelled) # label=label
	plt.xlabel('10 Nearest Crashes')
	plt.ylabel('Average distance travelled until crash')
	plt.title("Average distance travelled over time")
	#plt.legend()
	
	averageLoss = []
	i = 0
	accumLoss = 0
	try:
		while True:
			accumLoss += lossPerEpoch.popleft()
			i += 1
			if (i == 5):
				averageLoss.append(round(accumLoss / 5, 2))
				i = 0
				accumLoss = 0
	
	except IndexError:
		if (i != 0):
			averageLoss.append(round(accumLoss / i, 2))
	
	averageLoss = np.array(averageLoss)
	epochs = np.arange(1, len(averageLoss) + 1, 1)
	plt.figure(2)
	plt.plot(epochs, averageLoss) # label=label
	plt.xlabel('Epoch #')
	plt.ylabel('MSE Loss')
	plt.title("Loss over epochs")
	#plt.legend()
	
	averageEstimates = []
	i = 0
	accumEstim = 0
	try:
		while True:
			accumEstim += averageQvalueEstimates.popleft()
			i += 1
			if (i == 5):
				averageEstimates.append(round(accumEstim / 5, 2))
				i = 0
				accumEstim = 0
	
	except IndexError:
		if (i != 0):
			averageEstimates.append(round(accumEstim / i, 2))
	
	averageEstimates = np.array(averageEstimates)
	epochs = np.arange(1, len(averageEstimates) + 1, 1)
	plt.figure(3)
	plt.plot(epochs, averageEstimates)  # label=label
	plt.xlabel('Epoch #')
	plt.ylabel('Q Value Estimates')
	plt.title("Average Q value estimates per epoch")
	# plt.legend()
	
	if (len(game.racetrack.crashesPerLap) > 2):
		plt.figure(4)
		game.racetrack.crashesPerLap.pop(0)  # first number is not representative of performance and messes the scale of the graph
		crashes = np.array(game.racetrack.crashesPerLap)
		laps = np.arange(1, len(crashes) + 1, 1)
		plt.plot(laps, crashes) # label=label
		plt.xlabel('Lap #')
		plt.ylabel('Number of Crashes')
		plt.title("Crashes per completed lap")
		#plt.legend()
	plt.show()
	
	### session_doubleQ_6action
	### session_doubleQ_6action_(1024,512)
	#### session_doubleQ_4actions(Slow+MinSpeed=1)     very good
	# session_doubleQ_6actions(Slow+MinSpeed=1)_improved
	#### session_doubleQ_4actions(Slow+MinSpeed=1)_(512,256)
	###### session_duelingDQN_6actions(Slow+MinSpeed=1)_(1024,512)     GOOD BASE
	###### session_duelingDQN_6actions(Slow+MinSpeed=1)_(1024,512)_improved
	######### session_duelingDQN_6actions(Slow+MinSpeed=1)_(1024,512)_(e=0.01)    or _improved
	############ session_duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_(e=0.02)   BETTER BASE LESS OVERFIT
	'''
	# load a trained network
	saver.restore(sess, "./checkpoints/session_duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_(e=0.02).ckpt")
	while True:
		state = game.get_state()
		#print(state)
		Qvals = sess.run(mainQN.predictions, feed_dict={mainQN.inputs: np.atleast_2d(state)})
		#print(Qvals)
		action = np.argmax(Qvals, axis=1)[0]
		# Perform selected action in the game and get the reward
		game.perform_action(action)
		if (game.quit):
			break
	
	'''

''' actions:
0 - brake
1 - turn left
2 - turn right
3 - accelerate
4 - accelerate + turn left
5 - accelerate + turn right
'''