import pygame

class Racetrack():
	def __init__(self, setup):
		if (setup == "train" or setup == "test0"):
			self.prettyLayer = pygame.image.load("./images/trainTrack/pretty_layer.png").convert_alpha()
			self.finishLine = FinishLine((260, 615), 55)
			self.rewardLines = [RewardLine((180, 450), 90), RewardLine((240, 300), 90), RewardLine((130, 190), 0), RewardLine((40, 110), 90),
			                    RewardLine((150, 30), 0), RewardLine((420, 90), 0), RewardLine((440, 300), 90), RewardLine((520, 445), 0),
			                    RewardLine((740, 310), 0), RewardLine((1040, 260), 90), RewardLine((850, 115), 90), RewardLine((1070, 50), 0),
			                    RewardLine((1270, 250), 90), RewardLine((1070, 430), 90), RewardLine((1160, 660), 0), RewardLine((900, 540), 0), RewardLine((590, 590), 0)]
		elif (setup == "test1"):
			self.prettyLayer = pygame.image.load("./images/testTrack/1/pretty_layer.png").convert_alpha()
			self.finishLine = FinishLine((640, 640), 90)
			self.rewardLines = [RewardLine((1000, 650), 0)]
		elif (setup == "test2"):
			self.prettyLayer = pygame.image.load("./images/testTrack/2/pretty_layer.png").convert_alpha()
			self.finishLine = FinishLine((290, 630), 90)
			self.rewardLines = [RewardLine((700, 630), 0)]
		elif (setup == "test3"):
			self.prettyLayer = pygame.image.load("./images/testTrack/3/pretty_layer.png").convert_alpha()
			self.finishLine = FinishLine((565, 650), 90)
			self.rewardLines = [RewardLine((320, 650), 0)]
			
		self.road = Road(setup)
		self.roadGroup = pygame.sprite.Group(self.road)
		self.finishLineGroup = pygame.sprite.Group(self.finishLine)
		
		self.rewardLines[-1].last = True
		self.rewardLineGroup = pygame.sprite.Group(self.rewardLines)
		self.bestLapTime = 1000000
		self.lastLapTime = 0
		self.lastResetTime = 0
		self.completedLaps = 0
		self.numCrashes = 0  # number of crashes before finishing a lap
		self.crashesPerLap = []
	
	def record_lap(self, milliseconds):
		self.completedLaps += 1
		self.crashesPerLap.append(self.numCrashes)
		self.lastLapTime = round(milliseconds / 1000 - self.lastResetTime, 1)
		
		if (self.lastLapTime < self.bestLapTime):
			self.bestLapTime = self.lastLapTime
		
		self.reset(milliseconds)
		self.numCrashes = 0
		
	def reset(self, resetTime):
		self.numCrashes += 1
		self.finishLine.active = False
		self.lastResetTime = round(resetTime / 1000, 1)
		self.rewardLineGroup.update(resetTime)

class Road(pygame.sprite.Sprite):
	def __init__(self, setup):
		pygame.sprite.Sprite.__init__(self)
		if (setup == "train" or setup == "test0"):
			self.image = pygame.image.load("./images/trainTrack/road_edges.png").convert_alpha()
		elif (setup == "test1"):
			self.image = pygame.image.load("./images/testTrack/1/road_edges.png").convert_alpha()
		elif (setup == "test2"):
			self.image = pygame.image.load("./images/testTrack/2/road_edges.png").convert_alpha()
		elif (setup == "test3"):
			self.image = pygame.image.load("./images/testTrack/3/road_edges.png").convert_alpha()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect()
		self.rect.x = 0
		self.rect.y = 0

class FinishLine(pygame.sprite.Sprite):
	def __init__(self, position, rotation):
		pygame.sprite.Sprite.__init__(self)
		self.srcImage = pygame.image.load("./images/finish_line.png").convert_alpha()
		self.image = pygame.transform.rotate(self.srcImage, rotation)
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect()
		self.rect.center = position
		self.active = False

class RewardLine(pygame.sprite.Sprite):
	def __init__(self, position, rotation):
		pygame.sprite.Sprite.__init__(self)
		self.srcImage = pygame.image.load("./images/reward_line.png").convert()
		self.image = pygame.transform.rotate(self.srcImage, rotation)
		self.rect = self.image.get_rect()
		self.rect.center = position
		self.active = True
		self.last = False
		self.timeCrossed = 0
		self.lastResetTime = 0
	
	def update(self, resetTime):
		self.active = True
		self.timeCrossed = 0
		self.lastResetTime = round(resetTime / 1000, 1)
	
	def disable(self, milliseconds):
		self.active = False
		self.timeCrossed = round(milliseconds / 1000 - self.lastResetTime, 1)
		# if this is the last reward line, activate finish line
		if (self.last):
			return True
		return False