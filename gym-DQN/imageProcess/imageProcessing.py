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
	minBGR = np.array([bgr_list[0] - thresh, bgr_list[1] - thresh, bgr_list[2] - thresh])
	maxBGR = np.array([bgr_list[0] + thresh, bgr_list[1] + thresh, bgr_list[2] + thresh])

	mask = cv2.inRange(frame, minBGR, maxBGR)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.boxPoints(rect))

	# visualization, draw a bounding box around the detected barcode and display the image
	# cv2.drawContours(closing, [box], -1, (255, 255, 255), 3)
	# cv2.imshow("Image", closing)
	# cv2.waitKey(0)

	return box


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"sample2.png")

	POLE_COLOR = [102, 153, 204]
	THRESH = 40

	# cropped = crop_frame(img)
	coordinates = extract_objects(img, POLE_COLOR, THRESH)
	print(coordinates)
	print(type(coordinates))
