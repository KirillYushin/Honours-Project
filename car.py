import pygame
import math
from radar import Radar

class Car(pygame.sprite.Sprite):
	def __init__(self, startingPosition, startingRotation, setup, ai):
		pygame.sprite.Sprite.__init__(self)
		self.startingPosition = startingPosition
		self.startingRotation = startingRotation
		self.ai = ai
		self.x = startingPosition[0]
		self.y = startingPosition[1]
		self.acceleration = 1.0
		self.speed = 0
		self.direction = startingRotation
		self.maxSpeed = 10
		self.turnRate = 5
		self.srcImage = pygame.image.load('./images/car.png').convert_alpha()
		self.image = self.srcImage
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect()
		self.rect.center = (self.x, self.y)
		self.radar = Radar(setup)
	
	def draw(self, display):
		display.blit(self.image, self.rect)
	
	def update(self):
		if (self.ai):
			self.speed = round(self.speed * 0.95, 1)    # slow down if not accelerating
			
		if (self.speed > self.maxSpeed):
			self.speed = self.maxSpeed
		elif (self.speed < self.acceleration):
			if (self.ai):
				self.speed = 1.0
			else:
				self.speed = 0
		
		if (self.direction >= 360):
			self.direction -= 360
		elif (self.direction <= -360):
			self.direction += 360
		
		# Update car position
		radians = self.direction * math.pi / 180
		self.x += -self.speed * math.sin(radians)
		self.y += -self.speed * math.cos(radians)
		if (not self.ai):
			self.speed = round(self.speed * 0.95, 1)    # slow down if not accelerating
		self.image = pygame.transform.rotate(self.srcImage, self.direction)
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect()
		self.rect.center = (self.x, self.y)
		
		# Update Radar
		self.radar.update(self.x, self.y, self.direction)
		
	# Reset car and place it at the start
	def crash(self):
		self.x = self.startingPosition[0]
		self.y = self.startingPosition[1]
		self.speed = 0
		self.direction = self.startingRotation
		self.update()