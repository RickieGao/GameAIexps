import cv2
import numpy as np


def crop_frame(frame):
	"""
	function: crop a image
	:param frame: a image to be cropped
	:return: the cropped image
	"""
	return frame[35:160, 8:165]


def extract_objects(frame, bgr_list, thresh):
	"""
	function: extract the object specified color
	:param frame: image to be processing
	:param bgr_list: BGR color of objects to be extracted
	:param thresh: range of color
	:return: a list of coordinates of the center of extracted objects, just like [()]
	"""
	# color range of object
	minBGR = np.array([bgr_list[0] - thresh, bgr_list[1] - thresh, bgr_list[2] - thresh])
	maxBGR = np.array([bgr_list[0] + thresh, bgr_list[1] + thresh, bgr_list[2] + thresh])

	mask = cv2.inRange(frame, minBGR, maxBGR)
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	coordinate_list = []
	for cnt in cnts:
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		coordinate_list.append(center)

	# visualization
	# cv2.imshow("Result", mask)
	# cv2.waitKey()

	return coordinate_list


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"E:\Reserch\GameAI\skier\image_processing_lab\sample2.png")

	PLAYER_COLOR = [40, 240, 255]
	FLAG_COLOR = [1, 1, 255]
	TREE_COLOR = [1, 255, 1]

	THRESH = 40

	# cropped = crop_frame(img)
	coordinates = extract_objects(img, TREE_COLOR, THRESH)
	print(coordinates)
	print(len(coordinates))
