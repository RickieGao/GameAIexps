import cv2
from random import choice
import imageProcessing

POLE_COLOR = [102, 153, 204]
THRESH = 40


def rule_action(frame):
	"""
	function: according to the incline orientation of the pole return the action based rule
	:param frame: the input frame
	:return: the action based rule
	"""
	box = imageProcessing.extract_objects(frame, POLE_COLOR, THRESH)

	# numpy array to list
	box_list = box.tolist()
	# sorted according to Y coordinate
	box_list = sorted(box_list, key=lambda x: x[1])
	if box_list[0][0] >= box_list[2][0]:
		return "right"
	else:
		return "left"


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"image_process_lab/sample1.png")
	# cv2.imshow("test", img)
	# cv2.waitKey()
	action = rule_action(img)
	print(action)

