import cv2
import numpy as np
import time

def ExtractBird(frame):
	# range of red color in HSV
	lower_red = np.array([0,43,46])
	upper_red = np.array([10,255,255])
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_red, upper_red)
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	ret = []
	if len(cnts) == 1:
		((x, y), radius) = cv2.minEnclosingCircle(cnts[0])
		x = int(x)
		y = int(y)
		center = (x, y)
		ret.append(center)
		ret = ret + [(x-15, y-13), (x-15, y+13), (x+22, y+13), (x+22,y-13)]		# fixed value

	return ret

def ExtractPipe(frame):
	# range of green color in HSV
	lower_green = np.array([35, 43, 46])
	upper_green = np.array([77, 255, 255])

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_green, upper_green)
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	rets = []
	for cnt in cnts:															# cnt is a numpy array of coordinates
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		x = int(x)
		y = int(y)
		center = (x, y)
		ret = [center]
		for vertex in cnt:														# vertex is [[x, y]]
			ret.append(tuple(vertex[0]))
		rets.append(ret)

	return rets

def ExtractGround(frame):
	# range of yellow color in HSV
	lower_yellow = np.array([26, 43, 46])
	upper_yellow = np.array([34, 255, 255])

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	ret = []
	if len(cnts) == 1:
		cnt = cnts[0]
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		x = int(x)
		y = int(y)
		center = (x, y)
		ret.append(center)
		for vectex in cnt:
			ret.append(tuple(vectex[0]))

	return ret


if __name__ == '__main__':
	frame = cv2.imread('sample3.png')
	'''
	bird_points = ExtractBird(frame)
	pipes_points = ExtractPipe(frame)
	ground_points = ExtractGround(frame)

	draw_outline = lambda v1, v2 : cv2.rectangle(frame, v1, v2, (255, 0, 0), 1)
	draw_center = lambda center : cv2.circle(frame, center, 2, (255, 0, 0), -1)
	
	draw_outline(bird_points[1], bird_points[3])
	draw_center(bird_points[0])
	draw_outline(ground_points[1], ground_points[3])
	draw_center(ground_points[0])
	for pipe_points in pipes_points:
		draw_outline(pipe_points[1], pipe_points[3])
		draw_center(pipe_points[0])
	
	cv2.imshow('frame', frame)

	if cv2.waitKey(0) == ord('q'):
		cv2.destroyAllWindows()
	'''
	original_frame = frame.copy()
	bird_points = ExtractBird(frame)
	vertex1, vertex2 = bird_points[1], bird_points[3]
	v1_x, v1_y = vertex1[0], vertex1[1]
	v2_x, v2_y = vertex2[0], vertex2[1]
	bird_roi = frame[v1_y : v2_y, v1_x : v2_x]
	bird = bird_roi.copy()									# necessary to make a copy from the frame

	offset = 10
	frame[v1_y : v2_y, v1_x : v2_x] = (0, 0, 0)				# clean the area of the frame
	frame[v1_y + offset : v2_y + offset, v1_x + offset : v2_x + offset] = bird		# copy the bird to the new position
	cv2.imshow('bird', bird)
	cv2.imshow('after', frame)								# after
	cv2.imshow('before', original_frame)					# before

	if cv2.waitKey(0) == ord('q'):
		cv2.destroyAllWindows()