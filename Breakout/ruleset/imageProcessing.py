import cv2
import numpy as np


def extract_objects(frame):
	"""
	function: extract the objects
	:param frame: image to be processing
	:return: a list of coordinates of the center and the radius of extracted objects, just like [((x,y),r)]
	"""
	# minBGR = np.array([bgr_list[0] - thresh, bgr_list[1] - thresh, bgr_list[2] - thresh])
	# maxBGR = np.array([bgr_list[0] + thresh, bgr_list[1] + thresh, bgr_list[2] + thresh])
	#
	# mask = cv2.inRange(frame, minBGR, maxBGR)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
	cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	objects = []
	for cnt in cnts:
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		objects.append((center, int(radius)))

	# visualization
	# cv2.imshow("Image", mask)
	# cv2.waitKey(0)

	return objects


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"imageExtractionLab/sample6.png")

	coordinates = extract_objects(img)
	print(coordinates)
	print(len(coordinates))
