import pygame
from game import Game

class Menu:
	def __init__(self):
		pygame.init()
		pygame.display.set_caption("Racing game")
		self.clock = pygame.time.Clock()
		self.screenWidth = 1400
		self.screenHeight = 700
		self.gameDisplay = pygame.display.set_mode((self.screenWidth, self.screenHeight))
		self.quit = False

		self.btn_traintrack = pygame.Rect(150, 200, 200, 50)
		self.btn_testtrack_easy = pygame.Rect(450, 200, 200, 50)
		self.btn_testtrack_medium = pygame.Rect(750, 200, 200, 50)
		self.btn_testtrack_hard = pygame.Rect(1050, 200, 200, 50)
		self.track_btn_colors = [(170, 175, 180), (170, 175, 180), (170, 175, 180), (170, 175, 180)]
		self.selectedTrack = -1
		
		self.btn_nn_on = pygame.Rect(800, 350, 80, 50)
		self.btn_nn_off = pygame.Rect(880, 350, 80, 50)
		self.nn_btn_colors = [(50, 200, 50), (170, 175, 180)]
		self.enableNN = True
		self.btn_start = pygame.Rect(630, 500, 120, 50)

	def display_text(self, text, size,  x, y):
		font = pygame.font.SysFont("arial", size)
		text_rendered = font.render(text, True, (0, 0, 0))
		self.gameDisplay.blit(text_rendered, (x, y))

	def draw(self):
		self.gameDisplay.fill((255, 255, 255))
		self.display_text("Select a Racetrack", 40, 550, 100)
		pygame.draw.rect(self.gameDisplay, self.track_btn_colors[0], self.btn_traintrack)
		pygame.draw.rect(self.gameDisplay, self.track_btn_colors[2], self.btn_testtrack_easy)
		pygame.draw.rect(self.gameDisplay, self.track_btn_colors[3], self.btn_testtrack_medium)
		pygame.draw.rect(self.gameDisplay, self.track_btn_colors[1], self.btn_testtrack_hard)
		self.display_text("Training", 30, 205, 205)
		self.display_text("Test Easy", 30, 500, 205)
		self.display_text("Test Medium", 30, 780, 205)
		self.display_text("Test Hard", 30, 1100, 205)
		
		self.display_text("Neural Network:", 40, 550, 350)
		pygame.draw.rect(self.gameDisplay, self.nn_btn_colors[0], self.btn_nn_on)
		pygame.draw.rect(self.gameDisplay, self.nn_btn_colors[1], self.btn_nn_off)
		self.display_text("ON", 30, 820, 355)
		self.display_text("OFF", 30, 900, 355)
		
		pygame.draw.rect(self.gameDisplay, (25, 200, 10), self.btn_start)
		self.display_text("Start", 40, 650, 500)
		
		pygame.display.update()

	def select_track(self, trackId):
		self.selectedTrack = trackId
		self.track_btn_colors = [(170, 175, 180), (170, 175, 180), (170, 175, 180), (170, 175, 180)]
		self.track_btn_colors[trackId] = (50, 200, 50)

	def select_nn_state(self, state):
		self.enableNN = state
		self.nn_btn_colors = [(170, 175, 180), (170, 175, 180)]
		if (state):
			self.nn_btn_colors[0] = (50, 200, 50)
		else:
			self.nn_btn_colors[1] = (230, 5, 5)
			
	
	def run(self):
		while (True):
			self.clock.tick(60)
			mx, my = pygame.mouse.get_pos()
			click = False
			
			for event in pygame.event.get():
				if (event.type == pygame.QUIT):
					self.quit = True
				elif (event.type == pygame.KEYDOWN):
					if (event.key == pygame.K_q):
						self.quit = True
				elif (event.type == pygame.MOUSEBUTTONDOWN):
					if (event.button == 1):
						click = True
			
			if (self.quit):
				pygame.quit()
				break
			
			if (self.btn_traintrack.collidepoint(mx, my)):
				if (click):
					self.select_track(0)
			if (self.btn_testtrack_easy.collidepoint(mx, my)):
				if (click):
					self.select_track(2)
			if (self.btn_testtrack_medium.collidepoint(mx, my)):
				if (click):
					self.select_track(3)
			if (self.btn_testtrack_hard.collidepoint(mx, my)):
				if (click):
					self.select_track(1)
			if (self.btn_nn_on.collidepoint(mx, my)):
				if (click):
					self.select_nn_state(True)
			if (self.btn_nn_off.collidepoint(mx, my)):
				if (click):
					self.select_nn_state(False)
			if (self.btn_start.collidepoint(mx, my)):
				if (click and self.selectedTrack != -1):
					pygame.quit()
					break
			
			self.draw()

mainMenu = Menu()
mainMenu.run()

if (not mainMenu.quit):
	gameSetup = "test" + str(mainMenu.selectedTrack)
	game = Game(gameSetup, mainMenu.enableNN)
	game.run()