import cv2
import numpy as np


def crop_frame(frame):
	"""
	function: crop a image
	:param frame: a image to be cropped
	:return: the cropped image
	"""
	return frame[35:160, 8:165]


def PygameSurfaceToCV2Frame(pygame_surface):
	'''
		Convert pygame surface to cv2 frame.
	'''
	frame = cv2.transpose(pygame_surface)											# swap X and Y, which makes it readable
	# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	return frame


def extract_objects(frame, bgr_list, thresh):
	"""
	function: extract the object specified color
	:param frame: image to be processing
	:param bgr_list: BGR color of objects to be extracted
	:param thresh: range of color
	:return: a list of coordinates of the center of extracted objects, just like [()]
	"""
	frame = PygameSurfaceToCV2Frame(frame)
	# np.save("ok.npy", frame)
	minBGR = np.array([bgr_list[0] - thresh, bgr_list[1] - thresh, bgr_list[2] - thresh])
	maxBGR = np.array([bgr_list[0] + thresh, bgr_list[1] + thresh, bgr_list[2] + thresh])

	mask = cv2.inRange(frame, minBGR, maxBGR)
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	coordinate_list = []
	for cnt in cnts:
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		coordinate_list.append(center)

	return coordinate_list


# test sample
if __name__ == '__main__':
	img = np.load("ok.npy")
	# img = cv2.imread("ok.png")

	PLAYER_COLOR = [41, 212, 56]
	BULLET_COLOR = [81, 234, 234]
	ENEMY_COLOR = [61, 106, 242]
	THRESH = 40

	# cropped = crop_frame(img)
	coordinates = extract_objects(img, ENEMY_COLOR, THRESH)
	print(coordinates)
	print(len(coordinates))
	cv2.imshow("img", img)
	cv2.waitKey(0)
