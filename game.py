import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set the position of the window
os.environ['SDL_VIDEO_WINDOW_POS'] = "10,40"
import tensorflow.compat.v1 as tf
import pygame
import numpy as np
from car import Car
from racetrack import Racetrack
from nn import NeuralNetwork

class Game:
	def __init__(self, setup, ai):
		pygame.init()
		pygame.display.set_caption("Racing game")
		self.setup = setup
		self.ai = ai
		self.quit = False
		self.font = pygame.font.SysFont("arial", 20)
		self.clock = pygame.time.Clock()
		self.screenWidth = 1400
		self.screenHeight = 700
		self.gameDisplay = pygame.display.set_mode((self.screenWidth, self.screenHeight))
		self.showRadar = False
		self.showInfo = False
		self.simplifyRoad = False
		self.drawDisplay = True
		if (setup == "train" or setup == "test0"):
			self.car = Car((230, 580), 55, setup, ai)
		elif (setup == "test1"):
			self.car = Car((580, 620), 90, setup, ai)
		elif (setup == "test2"):
			self.car = Car((400, 605), 90, setup, ai)
		elif (setup == "test3"):
			self.car = Car((110, 350), 0, setup, ai)
		
		self.racetrack = Racetrack(setup)
		self.maxPoints = np.sum(self.car.radar.beamLength) + (self.car.maxSpeed * 20)
		self.car.update()
		
		if (ai and "test" in setup):
			self.mainQN = NeuralNetwork(self.car.radar.numBeams + 1, 6, "main")
			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.saver.restore(self.sess, "./checkpoints/session_duelingDQN_4actions(Slow+MinSpeed=1)_(1024,512)_150k.ckpt")
			# "./checkpoints/session_duelingDQN_6actions(Slow+MinSpeed=1)_(512,256)_(e=0.02).ckpt")
	
	def display_text(self, text, x, y):
		text_rendered = self.font.render(text, True, (0, 0, 0))
		self.gameDisplay.blit(text_rendered, (x, y))
	
	# Loop through events happened in the frame
	def handle_events(self, NNaction=-1):
		for event in pygame.event.get():
			if (event.type == pygame.QUIT):
				self.quit = True
			elif (event.type == pygame.KEYDOWN):
				if (event.key == pygame.K_q):
					self.quit = True
				elif (event.key == pygame.K_r):
					self.showRadar = not self.showRadar
				elif (event.key == pygame.K_i):
					self.showInfo = not self.showInfo
				elif (event.key == pygame.K_s):
					self.simplifyRoad = not self.simplifyRoad
				elif (event.key == pygame.K_d):
					self.drawDisplay = not self.drawDisplay
		
		# Car controls
		if(self.ai):  # By AI
			if (NNaction >= 3):
				self.car.speed += self.car.acceleration
			elif (NNaction == 0):
				self.car.speed -= self.car.acceleration / 2
			if ((NNaction == 1 or NNaction == 4) and self.car.speed >= self.car.acceleration):
				self.car.direction += self.car.turnRate
			elif ((NNaction == 2 or NNaction == 5) and self.car.speed >= self.car.acceleration):
				self.car.direction -= self.car.turnRate
		else:  # By human
			keys = pygame.key.get_pressed()
			if (keys[pygame.K_UP]):
				self.car.speed += self.car.acceleration
			elif (keys[pygame.K_DOWN]):
				self.car.speed -= self.car.acceleration / 2
			if (keys[pygame.K_LEFT] and self.car.speed >= self.car.acceleration):
				self.car.direction += self.car.turnRate
			elif (keys[pygame.K_RIGHT] and self.car.speed >= self.car.acceleration):
				self.car.direction -= self.car.turnRate
	
	# Redraw everything on the display
	def draw(self):
		self.gameDisplay.fill((100, 100, 100))
		if (self.setup == "train" or self.setup == "test0"):
			self.racetrack.finishLineGroup.draw(self.gameDisplay)
		
		# Redraw radar
		if (self.showRadar):
			for i in range(self.car.radar.numBeams):
				# radar beams
				pygame.draw.line(self.gameDisplay, (255, 0, 0), self.car.radar.beamStart, self.car.radar.beamEnd[i], 2)
				# points of intersection with the road edges
				pygame.draw.circle(self.gameDisplay, (0, 255, 0), self.car.radar.beamPoint[i], 10)
		
		# Pretty or simplified view of the Racetrack
		if (self.simplifyRoad):
			self.racetrack.roadGroup.draw(self.gameDisplay)
			self.racetrack.rewardLineGroup.draw(self.gameDisplay)
			self.racetrack.finishLineGroup.draw(self.gameDisplay)
		else:
			self.gameDisplay.blit(self.racetrack.prettyLayer, (0, 0))
		
		self.car.draw(self.gameDisplay)
		
		if (self.showInfo):
			self.display_text("Speed: " + str(self.car.speed), 10, 10)
			self.display_text("Crashes: " + str(self.racetrack.numCrashes), 10, 30)
			self.display_text("Laps: " + str(self.racetrack.completedLaps), 10, 50)
			self.display_text("Last lap time: " + str(self.racetrack.lastLapTime), 10, 70)
			self.display_text("Best lap time: " + str(self.racetrack.bestLapTime), 10, 90)
			
			#self.display_text("Mouse: " + str(pygame.mouse.get_pos()), 1100, 10)
			# display distances
			for i in range(self.car.radar.numBeams):
				self.display_text("Dist" + str(i + 1) + ": " + str(self.car.radar.beamDist[i]), 1300, 10 + 20 * i)
			#for j in range(len(self.racetrack.rewardLines)):
				#self.display_text("Line" + str(j + 1) + ": " + str(self.racetrack.rewardLines[j].timeCrossed), 1300, 120 + 20 * j)
		
		'''
		# Road split into individual lines that are used to find intersections with radar beams
		for i in range(1, len(self.car.radar.pointList)):
			if (self.car.radar.pointList[i] not in self.car.radar.excludedPoints):
				pygame.draw.line(self.gameDisplay, (0, 0, 0), self.car.radar.pointList[i-1], self.car.radar.pointList[i], 2)
		'''
		
		pygame.display.update()
		
	# Main game loop
	def run(self):
		while (True):
			self.clock.tick(60)  # set fps
			# If AI is playing, select action from trained Neural Network and handle all events
			if (self.ai):
				state = self.get_state()
				action = self.sess.run(self.mainQN.predict_op, feed_dict={self.mainQN.inputs: np.atleast_2d(state)})[0]
				self.handle_events(action)
			# If human is playing
			else:
				self.handle_events()
				
			if (self.quit):
				pygame.quit()
				if (self.ai):
					self.sess.close()
				break
				
			self.car.update()
			
			# Check collision with the track
			collisions = pygame.sprite.spritecollide(self.car, self.racetrack.roadGroup, False, pygame.sprite.collide_mask)
			if (len(collisions) != 0):
				self.car.crash()
				self.racetrack.reset(pygame.time.get_ticks())
				
			# Check collision with the finish line
			elif (self.racetrack.finishLine.active):
				finishLineCollision = pygame.sprite.spritecollide(self.car, self.racetrack.finishLineGroup, False, pygame.sprite.collide_mask)
				if (len(finishLineCollision) != 0):
					self.racetrack.record_lap(pygame.time.get_ticks())
					
			# Check collisions with reward lines
			else:
				rewardLineCollisions = pygame.sprite.spritecollide(self.car, self.racetrack.rewardLineGroup, False)
				for line in rewardLineCollisions:
					if (line.active):
						# if this is the last reward line, activate finish line
						if (line.disable(pygame.time.get_ticks())):
							self.racetrack.finishLine.active = True

			self.draw()
	
	
	### Functions for the Neural Network ###
	
	# Get current state of the game (list of radar distances and current car speed)
	# Adjusted to be in the range from 0 to 1. Values represent proportions of their max possible values
	def get_state(self):
		state = np.array(self.car.radar.beamDist) / self.car.radar.beamLength
		state = np.append(state, self.car.speed / self.car.maxSpeed)
		return np.around(state, decimals=2)
	
	# Perform action selected by NN during training and update the game
	# Returns reward
	def perform_action(self, action):
		self.handle_events(action)
		if (self.quit):
			pygame.quit()
			return 0
		
		self.car.update()
		
		# Custom reward function
		if (self.car.speed > 0):
			# Award points based on how the action maximises distances and car speed
			collectedPoints = np.sum(self.car.radar.beamDist) + (self.car.speed * 25)
			reward = round((collectedPoints / self.maxPoints) * 20, 2)
		else:
			# Punish if action did not move the car or made it stop
			reward = -10
		
		# Check collision with the track
		collisions = pygame.sprite.spritecollide(self.car, self.racetrack.roadGroup, False, pygame.sprite.collide_mask)
		if (len(collisions) != 0):
			self.car.crash()
			self.racetrack.reset(pygame.time.get_ticks())
			reward = -200
		
		# Check collision with the finish line
		elif (self.racetrack.finishLine.active):
			finishLineCollision = pygame.sprite.spritecollide(self.car, self.racetrack.finishLineGroup, False, pygame.sprite.collide_mask)
			if (len(finishLineCollision) != 0):
				self.racetrack.record_lap(pygame.time.get_ticks())
				#reward = 200
		
		# Check collisions with reward lines
		else:
			rewardLineCollisions = pygame.sprite.spritecollide(self.car, self.racetrack.rewardLineGroup, False)
			for line in rewardLineCollisions:
				if (line.active):
					#reward = 200
					# if this is the last reward line, activate finish line
					if (line.disable(pygame.time.get_ticks())):
						self.racetrack.finishLine.active = True
		
		if (self.drawDisplay):
			self.draw()
		return reward

if __name__ == "__main__":
	game = Game("test3", True)
	game.run()
