#!/usr/bin python

'''
	All about image processing.
'''
import sys
sys.path.append("ruleset/")
import cv2
from functools import cmp_to_key
from func import *					# Cooper: My image processing functions

def PygameSurfaceToCV2Frame(pygame_surface):
	'''
		Convert pygame surface to cv2 frame.
	'''
	frame = cv2.transpose(pygame_surface)											# swap X and Y, which makes it readable
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	return frame

def DigOutObjects(frame):
	'''
		This function extracts the positions of bird, pipes and ground,
		returning their coordinate sets respectively. Each coordinate
		set has 5 tuples: [(centroid), (A), (B), (C), (D)], where ABCD
		is the left-top, left-bottom, right-bottom and right-top vectex.

		'bird_points' and 'ground_points' contains coordinates of
		centroid and vertices of the minium enclosing rectangle.

		'pipes_points' is an array, in which every element is a coordinate
		set of a pipe.
	'''
	bird_points = ExtractBird(frame)
	pipes_points = ExtractPipe(frame)
	ground_points = ExtractGround(frame)

	return bird_points, pipes_points, ground_points

class Vertex:
	'''
		A vertex wrapper
	'''
	def __init__(self, coordinate_tuple):
		self.x = coordinate_tuple[0]
		self.y = coordinate_tuple[1]

	def GetX(self):
		return self.x

	def GetY(self):
		return self.y

'''
	The following 5 functions receive a coordinate set of an object,
	returning corresponding vertex.
'''
def GetCentroid(coordinate_set):
	return Vertex(coordinate_set[0])

def GetVertexA(coordinate_set):
	return Vertex(coordinate_set[1])

def GetVertexB(coordinate_set):
	return Vertex(coordinate_set[2])

def GetVertexC(coordinate_set):
	return Vertex(coordinate_set[3])

def GetVertexD(coordinate_set):
	return Vertex(coordinate_set[4])

def BirdOutofGap(bird, top_pipe, bottom_pipe):
	'''
		If the bird is higher than the pipe gap, returns 'H',
		if lower, returns 'L',
		otherwise, returns 'M'.
	'''
	bird_upper = GetVertexA(bird).GetY()
	bird_lower = GetVertexC(bird).GetY()
	gap_upper = GetVertexC(top_pipe).GetY()
	gap_lower = GetVertexA(bottom_pipe).GetY()
	if bird_upper < gap_upper:
		return 'H'
	elif bird_lower > gap_lower:
		return 'L'
	else:
		return 'M'

def BirdCrossPipe(bird, pipe):
	'''
		This function returns whether the bird has crossed the pipe.
	'''
	pipe_right_bound = GetVertexC(pipe).GetX()
	bird_left_bound = GetVertexA(bird).GetX()
	if bird_left_bound >= pipe_right_bound:
		return True
	else:
		return False

def SortPipes(pipes):
	'''
		pipes is an array of serveral pairs of coordinate sets of pipe.
		This function sorts pipes by their centroid coordinates,
		The pair of pipes near the bird will appear in front of the
		returning list, and the top pipe of the same pair of pipe will
		appear in front of the bottom pipe. 
	'''
	SortX = lambda a, b: GetCentroid(a).GetX() - GetCentroid(b).GetX()
	pipes.sort(key=cmp_to_key(SortX))
	# if the bottom pipe of the same pair of pipe is in front of the
	# top one, swap them.
	for i in range(len(pipes) // 2):
		if GetCentroid(pipes[i*2]).GetY() > GetCentroid(pipes[i*2+1]).GetY():
			pipes[i*2], pipes[i*2+1] = pipes[i*2+1], pipes[i*2]
	return pipes

def PipesToCross(bird, pipes):
	'''
		returns the nearest pair of pipes to cross.
	'''
	pipes = SortPipes(pipes)
	for i in range(len(pipes) // 2):
		cross = BirdCrossPipe(bird, pipes[i*2])
		if not cross:
			return pipes[i*2], pipes[i*2+1]
	return [], []

def AdviseAction(pygame_surface):
	'''
		This function will advise the action the bird take.
		Returning value 'U' means the bird should go up,
		'D' means the bird should go down,
		'N' means no advice action(the bird is safe),
		'E' means error.
	'''
	frame = PygameSurfaceToCV2Frame(pygame_surface)
	bird, pipes, _ = DigOutObjects(frame)
	if len(pipes) == 0:
		return 'N'

	top_pipe, bottom_pipe = PipesToCross(bird, pipes)
	if top_pipe == [] or bottom_pipe == []:
		return 'E'

	bird_pos = BirdOutofGap(bird, top_pipe, bottom_pipe)

	# if the bird is higher than gap, go down,
	# if lower, go up,
	# if just in gap, no advise,
	# othewise, error occurs.
	if bird_pos == 'H':
		return 'D'
	elif bird_pos == 'L':
		return 'U'
	elif bird_pos == 'M':
		return 'N'
	else:
		return 'E'

def ShowProfile(pygame_surface):
	'''
		Draw profiles of objects
	'''
	frame = PygameSurfaceToCV2Frame(pygame_surface)
	bird, pipes, ground = DigOutObjects(frame)
	draw_outline = lambda v1, v2 : cv2.rectangle(frame, v1, v2, (255, 0, 0), 1)
	draw_center = lambda center : cv2.circle(frame, center, 2, (255, 0, 0), -1)

	draw_outline(bird[1], bird[3])
	draw_center(bird[0])
	draw_outline(ground[1], ground[3])
	draw_center(ground[0])
	for pipe in pipes:
		draw_outline(pipe[1], pipe[3])
		draw_center(pipe[0])

	if len(pipes) >= 2:
		top, bottom = PipesToCross(bird, pipes)
		draw_line = lambda v1, v2 : cv2.line(frame, v1, v2, (0, 0, 255), 1)
		line1_x, line1_y = GetVertexC(top).GetX(), GetVertexC(top).GetY()
		draw_line((0, line1_y), (line1_x, line1_y))
		line2_x, line2_y = GetVertexD(bottom).GetX(), GetVertexD(bottom).GetY()
		draw_line((0, line2_y), (line2_x, line2_y))

	cv2.imshow('Profiles', frame)
	refresh_ms = 100
	cv2.waitKey(refresh_ms)

#---------------------------------------

def PredictFrame(pygame_surface, bird_points, delta_x, delta_y, title):
	frame = cv2.transpose(pygame_surface)											# swap X and Y, which makes it readable
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

	vertex1, vertex2 = bird_points[1], bird_points[3]
	v1_x, v1_y = vertex1[0], vertex1[1]
	v2_x, v2_y = vertex2[0], vertex2[1]
	bird_roi = frame[v1_y : v2_y, v1_x : v2_x]
	bird = bird_roi.copy()															# necessary to make a copy from the frame
	frame[v1_y : v2_y, v1_x : v2_x] = (0, 0, 0)										# clean the area of the frame
	if v1_y + delta_y < 0:
		# if the bird will flaps over the screen, keep the bird stay in the screen
		bird_height = v2_y - v1_y
		v1_y = 0
		v2_y = bird_height
		v1_x += delta_x
		v2_x += delta_x
		frame[v1_y : v2_y, v1_x : v2_x] = bird			# copy the bird to the new position
	else:
		v1_y += delta_y
		v2_y += delta_y
		v1_x += delta_x
		v2_x += delta_x
		frame[v1_y : v2_y, v1_x : v2_x] = bird			# copy the bird to the new position
	'''
	# Cooper: comment out to avoid drawing frames
	cv2.imshow(title, frame)
	refresh_ms = 100
	cv2.waitKey(refresh_ms)
	'''

	# print("%s, predict: (%d, %d)" % (title, bird_points[0][0] + delta_x, \
	#	bird_points[0][1] + delta_y))
	return frame

def GetPredictFrames(pygame_surface, bird_points, trace):
	'''
		Predict and draw both two frame of current state.
		We assume the direction of gravity is downward, thus, value of upward speed is negative.
		Caller should promise that "trace" argument is not empty.
	'''
	# 1, "Flap" case, go upward staight a little pixels (for example, 9):
	flap_frame = PredictFrame(pygame_surface, bird_points, 4, -9, 'Flap')

	# 2, "No flap" case, predict next position by history trace, we have to handle the terminal case carefully,
	# in which case the change of the bird's position looks obvious--after the crash, the bird will be reset
	# to the starting point, such movement trace will look abnormal once we consider the movement law of bird.
	#
	# The coordinate of last element of the trace is the current position of bird:
	last_shot = trace[-1]
	if last_shot.terminal:
		# If the game was terminal in last frame, the bird will be reset to the beginning,
		# "no flap" action makes the bird drop a little.
		delta_y = 1
	else:
		if last_shot.action == 1:
			# If last action is "flap", the bird continues going upward, pixels (for example, -8),
			# the bird cannot go out of the screen, the function "PredictFrame" will handle this situation
			delta_y = -8
		else:
			if len(trace) <= 1:
				# In the first round of game, the bird takes "no flap" action, the list "trace" has only one record,
				# this is the only case that the list "trace" has single record of which the action is "no flap"
				delta_y = 2
			else:
				# Go downward or upward, depending on history trace
				last_2nd_shot = trace[-2]
				last_2nd_shot_y = last_2nd_shot.center[1]
				last_shot_y = last_shot.center[1]
				difference = last_shot_y - last_2nd_shot_y
				if difference == 0:
					# begin to go down
					delta_y = 1
				elif difference == -1:
					# if "no flap", speed will become 0, the bird keeps the same height
					delta_y = 0
				elif difference < -1:
					# continue going upward, but slower
					delta_y = difference + 1
				else:
					# continue going down, faster
					delta_y = difference + 1
	noflap_frame = PredictFrame(pygame_surface, bird_points, 4, delta_y, 'No flap')
	return flap_frame, noflap_frame