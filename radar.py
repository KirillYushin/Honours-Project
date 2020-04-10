import pygame
import math

class Radar():
	def __init__(self, setup):
		# Convert image of road into list of points
		if (setup == "train" or setup == "test0"):
			self.pointList = pygame.mask.from_surface((pygame.image.load("./images/trainTrack/road_filled.png"))).outline(30)
			self.pointList.extend([(691, 577), (721, 556), (701, 441), (685, 450)])
			self.excludedPoints = [(701, 561), (701, 531), (701, 501), (701, 471), (701, 441), (698, 460), (698, 490), (698, 520), (698, 550), (691, 577)]
		elif (setup == "test1"):
			self.pointList = pygame.mask.from_surface((pygame.image.load("./images/testTrack/1/road_filled.png"))).outline(30)
			self.pointList.extend([(829, 27), (710, 557), (696, 557), (690, 669), (728, 669)])
			self.excludedPoints = [(700, 667), (700, 637), (700, 607), (700, 577), (710, 557), (697, 586), (697, 616), (697, 646), (690, 669)]
		elif (setup == "test2"):
			self.pointList = pygame.mask.from_surface((pygame.image.load("./images/testTrack/2/road_filled.png"))).outline(30)
			self.pointList.extend([(139, 42), (671, 652), (720, 652), (708, 556), (673, 556)])
			self.excludedPoints = [(699, 560), (699, 590), (699, 620), (699, 650), (671, 652), (702, 640), (702, 610), (702, 580), (708, 556)]
		elif (setup == "test3"):
			self.pointList = pygame.mask.from_surface((pygame.image.load("./images/testTrack/3/road_filled.png"))).outline(30)
			self.pointList.extend([(1013, 12), (713, 590), (697, 590), (675, 684), (717, 684)])
			self.excludedPoints = [(675, 684), (699, 618), (699, 648), (699, 678), (702, 669), (702, 639), (702, 609), (713, 590)]
			
		self.numBeams = 7
		# Length of each beam
		self.beamLength = [300.0, 250.0, 250.0, 150.0, 150.0, 100.0, 100.0]
		# Angle of each beam (from the main center beam)
		self.beamAngle = [0, 20, -20, 50, -50, 80, -80]
		# Beam start and end points
		self.beamStart = (0, 0)
		self.beamEnd = [(0, 0)] * self.numBeams
		# Intersection point of radar beam with the road
		self.beamPoint = [(0, 0)] * self.numBeams
		# Distance to intersection point
		self.beamDist = list(self.beamLength)
	
	
	def update(self, carX, carY, carDirection):
		# Update start and end points of every beam
		radians = carDirection * math.pi / 180
		self.beamStart = (int(carX + -30 * math.sin(radians)), int(carY + -30 * math.cos(radians)))
		for i in range(self.numBeams):
			radians = (carDirection + self.beamAngle[i]) * math.pi / 180
			self.beamEnd[i] = (int(self.beamStart[0] + -self.beamLength[i] * math.sin(radians)), int(self.beamStart[1] + -self.beamLength[i] * math.cos(radians)))
		
		# Find points where radar beams intersect with the road consisting of line segments and update distances to these points
		self.beamPoint = [(0, 0)] * self.numBeams
		self.beamDist = list(self.beamLength)
		for p in range(1, len(self.pointList)):
			if (self.pointList[p] not in self.excludedPoints):
				for b in range(self.numBeams):
					beamLine = [self.beamStart, self.beamEnd[b]]
					intersectPoint = segment_intersect_point(beamLine, (self.pointList[p - 1], self.pointList[p]))
					if (intersectPoint is not None):
						dist = distance(self.beamStart, intersectPoint)
						if (dist < self.beamDist[b]):
							self.beamPoint[b] = intersectPoint
							self.beamDist[b] = round(dist, 1)


# *** Radar related functions ***

'''
Base code taken from https://www.codeproject.com/Tips/864704/Python-Line-Intersection-for-Pygame
And modified by me (fixed bugs and added handling for vertical and parallel lines)
'''
# point is a tuple of x and y coordinates (x, y)
# the line 'data structure' looks like this:
# line = [(100,100),(700,700)]
# (defines a line segment between the points (100,100) and (700,700))

# Calculate distance between two points
def distance(p1, p2):
	dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
	return dist

# Calculate the slope 'm' of a line between p1 and p2
def slope(p1, p2):
	if (p1[0] != p2[0]):
		return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0])
	else:
		return None

# Calculate the point 'b' where line crosses the Y axis
def y_intercept(m, p1):
	return p1[1] - (1. * m * p1[0])


# Calculate the point where two infinitely long lines intersect.
# Handles parallel lines and vertical lines
# Returns a point tuple like this (x,y) or None
def intersect_point(line1, line2):
	min_allowed = 1e-4  # guard against overflow
	big_value = 1e4  # use instead if overflow occurred
	m1 = slope(line1[0], line1[1])
	m2 = slope(line2[0], line2[1])
	# See if the lines are parallel
	if (m1 != m2):
		# Not parallel
		
		# See if either line is vertical
		if (m1 is not None and m2 is not None):
			# Neither line vertical
			b1 = y_intercept(m1, line1[0])
			b2 = y_intercept(m2, line2[0])
			
			if (abs(m1 - m2) < min_allowed):
				x = big_value
			else:
				x = (b2 - b1) / (m1 - m2)
			y = m1 * x + b1
			return (int(x), int(y))
		else:
			# Line 1 is vertical so use line 2's values
			if (m1 is None):
				b2 = y_intercept(m2, line2[0])
				x = line1[0][0]
				y = (m2 * x) + b2
				return (int(x), int(y))
			# Line 2 is vertical so use line 1's values
			elif (m2 is None):
				b1 = y_intercept(m1, line1[0])
				x = line2[0][0]
				y = (m1 * x) + b1
				return (int(x), int(y))
	else:
		# Lines are parallel
		b1, b2 = None, None  # vertical lines have no b value
		if (m1 is not None):
			b1 = y_intercept(m1, line1[0])
		
		if (m2 is not None):
			b2 = y_intercept(m2, line2[0])
		
		# If these parallel lines lay on one another (same 'b' value)
		# In this case return the start point of the second line (in my case this will be enough)
		if (b1 == b2):
			if (b1 is None):  # case for 2 vertical lines
				if (line1[0][0] == line2[0][0]):  # 2 vertical lines lay on one another
					return line2[0]
				else:
					return None
			else:
				return line2[0]
		else:
			# Lines don't intersect
			return None
		
		
# Calculate the point where two line segments (not infinitely long lines) intersect
# Returns intersect point if the lines intersect or None if not
def segment_intersect_point(line1, line2):
	intersection_pt = intersect_point(line1, line2)
	
	if (intersection_pt is None):
		return None
	
	if (line1[0][0] < line1[1][0]):
		if (intersection_pt[0] < line1[0][0] or intersection_pt[0] > line1[1][0]):
			return None
	elif(line1[0][0] > line1[1][0]):
		if (intersection_pt[0] > line1[0][0] or intersection_pt[0] < line1[1][0]):
			return None
	else:
		# x-coordinates are equal, so vertical line. Check if intersection point is outside of line 1 segment
		if (line1[0][1] < line1[1][1]):
			if (intersection_pt[1] < line1[0][1] or intersection_pt[1] > line1[1][1]):
				return None
		else:
			if (intersection_pt[1] > line1[0][1] or intersection_pt[1] < line1[1][1]):
				return None
		
	
	if (line2[0][0] < line2[1][0]):
		if (intersection_pt[0] < line2[0][0] or intersection_pt[0] > line2[1][0]):
			return None
	elif(line2[0][0] > line2[1][0]):
		if (intersection_pt[0] > line2[0][0] or intersection_pt[0] < line2[1][0]):
			return None
	else:
		# x-coordinates are equal, so vertical line. Check if intersection point is outside of line 2 segment
		if (line2[0][1] < line2[1][1]):
			if (intersection_pt[1] < line2[0][1] or intersection_pt[1] > line2[1][1]):
				return None
		else:
			if (intersection_pt[1] > line2[0][1] or intersection_pt[1] < line2[1][1]):
				return None
	
	# Check for parallel vertical lines if they are in top of each other
	if (line1[0][0] == line1[1][0] and line2[0][0] == line2[1][0]):
		if (line1[0][1] < line1[1][1]):
			if (intersection_pt[1] < line1[0][1] or intersection_pt[1] > line1[1][1]):
				return None
		else:
			if (intersection_pt[1] > line1[0][1] or intersection_pt[1] < line1[1][1]):
				return None
	
	return intersection_pt
