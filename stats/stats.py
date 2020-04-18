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


# Laps

with open('./duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/crashesPerLap.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		crashesPerLap1 = list(row)
		break

with open('./doubleDQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/crashesPerLap.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		crashesPerLap2 = list(row)
		break

with open('./DQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/crashesPerLap.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		crashesPerLap3 = list(row)
		break

print(len(crashesPerLap1))
print(len(crashesPerLap2))
crashesPerLap1.pop(0)
crashesPerLap2.pop(0)
crashesPerLap3.pop(0)
crashesPerLap1 = np.array(crashesPerLap1, dtype=float)
crashesPerLap2 = np.array(crashesPerLap2, dtype=float)
crashesPerLap3 = np.array(crashesPerLap3, dtype=float)
laps1 = np.arange(1, len(crashesPerLap1) + 1, 1)
laps2 = np.arange(1, len(crashesPerLap2) + 1, 1)
laps3 = np.arange(1, len(crashesPerLap3) + 1, 1)

plt.figure(1)
plt.plot(laps1, crashesPerLap1, label="Dueling DQN")
plt.plot(laps2, crashesPerLap2, label="Double DQN")
plt.plot(laps3, crashesPerLap3, label="Regular DQN")
plt.xlabel('Lap #')
plt.ylabel('Number of Crashes')
plt.title("Crashes per completed lap")
plt.legend()
plt.show()

# Distance
'''
with open('./duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/distanceTravelledUntilCrash.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		distanceTravelledUntilCrash1 = collections.deque(row)
		break

with open('./doubleDQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/distanceTravelledUntilCrash.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		distanceTravelledUntilCrash2 = collections.deque(row)
		break

with open('./DQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/distanceTravelledUntilCrash.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		distanceTravelledUntilCrash3 = collections.deque(row)
		break

print(len(distanceTravelledUntilCrash1))
print(len(distanceTravelledUntilCrash2))
print(len(distanceTravelledUntilCrash3))

for i in range(500):
	distanceTravelledUntilCrash3.popleft()
for i in range(len(distanceTravelledUntilCrash1) - len(distanceTravelledUntilCrash3)):
	distanceTravelledUntilCrash1.popleft()
for i in range(len(distanceTravelledUntilCrash2) - len(distanceTravelledUntilCrash3)): # 
	distanceTravelledUntilCrash2.popleft()

averages1 = []
averages2 = []
averages3 = []
i = 0
k = 10
accum1 = 0
accum2 = 0
accum3 = 0
try:
	while True:
		accum1 += float(distanceTravelledUntilCrash1.popleft())
		accum2 += float(distanceTravelledUntilCrash2.popleft())
		accum3 += float(distanceTravelledUntilCrash3.popleft())
		i += 1
		if (i == k):
			averages1.append(round(accum1 / k, 2))
			averages2.append(round(accum2 / k, 2))
			averages3.append(round(accum3 / k, 2))
			i = 0
			accum1 = 0
			accum2 = 0
			accum3 = 0

except IndexError:
	if (i != 0):
		averages1.append(round(accum1 / i, 2))
		averages2.append(round(accum2 / i, 2))
		averages3.append(round(accum3 / i, 2))

averages1 = np.array(averages1)
averages2 = np.array(averages2)
averages3 = np.array(averages3)
x_axis = np.arange(1, len(averages1) + 1, 1)
plt.figure(1)
plt.plot(x_axis, averages1, label="Dueling DQN")
plt.plot(x_axis, averages2, label="Double DQN")
plt.plot(x_axis, averages3, label="Regular DQN")
plt.xlabel('10 Nearest Crashes')
plt.ylabel('Average distance travelled until crash')
plt.title("Average distance travelled over time")
plt.legend()
plt.show()
'''

# Loss
'''
with open('./Scaled Inputs but same length radar beams/lossPerEpoch.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		loss1 = collections.deque(row)
		break

with open('./Raw Inputs same length/lossPerEpoch.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		loss2 = collections.deque(row)
		break

print(len(loss1))
print(len(loss2))

for i in range(93):
	loss1.popleft()
for i in range(len(loss2) - len(loss1)):
	loss2.popleft()

averages1 = []
averages2 = []
i = 0
k = 5
accum1 = 0
accum2 = 0
try:
	while True:
		accum1 += float(loss1.popleft())
		accum2 += float(loss2.popleft())
		i += 1
		if (i == k):
			averages1.append(round(accum1 / k, 2))
			averages2.append(round(accum2 / k, 2))
			i = 0
			accum1 = 0
			accum2 = 0

except IndexError:
	if (i != 0):
		averages1.append(round(accum1 / i, 2))
		averages2.append(round(accum2 / i, 2))

averages1 = np.array(averages1)
averages2 = np.array(averages2)
x_axis = np.arange(1, len(averages1) + 1, 1)
plt.figure(1)
plt.plot(x_axis, averages1, label="Normalized inputs")
plt.plot(x_axis, averages2, label="Raw inputs")
plt.xlabel('Epoch #')
plt.ylabel('MSE Loss')
plt.title("Loss over epochs")
plt.legend()
plt.show()
'''
'''
# Q estimates
with open('./doubleDQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/averageQvalueEstimates.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		q1 = collections.deque(row)
		break

with open('./DQN_6actions(Slow+MinSpeed=1)_(512,256)_150k/averageQvalueEstimates.csv', 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		q2 = collections.deque(row)
		break


averageEstimates1 = []
averageEstimates2 = []
i = 0
accumEstim1 = 0
accumEstim2 = 0
try:
	while True:
		accumEstim1 += float(q1.popleft())
		accumEstim2 += float(q2.popleft())
		i += 1
		if (i == 5):
			averageEstimates1.append(round(accumEstim1 / 5, 2))
			averageEstimates2.append(round(accumEstim2 / 5, 2))
			i = 0
			accumEstim1 = 0
			accumEstim2 = 0

except IndexError:
	if (i != 0):
		averageEstimates1.append(round(accumEstim1 / i, 2))
		averageEstimates2.append(round(accumEstim2 / i, 2))

averageEstimates1 = np.array(averageEstimates1)
averageEstimates2 = np.array(averageEstimates2)
epochs = np.arange(1, len(averageEstimates1) + 1, 1)
plt.figure(1)
plt.plot(epochs, averageEstimates1, label="Double DQN")
plt.plot(epochs, averageEstimates2, label="Regular DQN")
plt.xlabel('Epoch #')
plt.ylabel('Q Value Estimates')
plt.title("Average Q value estimates per epoch")
plt.legend()
plt.show()
'''