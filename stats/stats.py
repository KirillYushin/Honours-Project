import numpy as np
import collections
import matplotlib.pyplot as plt
import csv
'''
with open('./distanceTravelledUntilCrash.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		distanceTravelledUntilCrash = collections.deque(row)
		break


averageDistanceTravelled = []
i = 0
accumDist = 0
try:
	while True:
		accumDist += float(distanceTravelledUntilCrash.popleft())
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


with open('./lossPerEpoch.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		lossPerEpoch = collections.deque(row)
		break


averageLoss = []
i = 0
accumLoss = 0
try:
	while True:
		accumLoss += float(lossPerEpoch.popleft())
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
plt.show()
'''

with open('./averageQvalueEstimates.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		averageQvalueEstimates = collections.deque(row)
		break

averageEstimates = []
i = 0
accumEstim = 0
try:
	while True:
		accumEstim += float(averageQvalueEstimates.popleft())
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
plt.figure(1)
plt.plot(epochs, averageEstimates)  # label=label
plt.xlabel('Epoch #')
plt.ylabel('Q Value Estimates')
plt.title("Average Q value estimates per epoch")
# plt.legend()
plt.show()