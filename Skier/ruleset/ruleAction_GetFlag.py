import cv2
from random import choice
import imageProcessing

PLAYER_COLOR = [1, 240, 255]
FLAG_COLOR = [1, 1, 255]
TREE_COLOR = [1, 255, 1]
THRESH = 40


def rule_action(frame):
	"""
	function: according to the relative location of closest enemy and player return the action based rule
	:param frame: the input frame
	:return: the action based rule
	"""
	player_coordinate = imageProcessing.extract_objects(frame, PLAYER_COLOR, THRESH)
	flag_coordinate = imageProcessing.extract_objects(frame, FLAG_COLOR, THRESH)

	# action options
	options = ["left", "right"]

	# if no enemy or more than one player, return random choice
	if len(flag_coordinate) == 0 or len(player_coordinate) > 1:
		return choice(options)
	else:
		# find the enemy that closest to player, record it at a tuple
		closest_flag = min(flag_coordinate, key=lambda x: x[1])

		# compare the X coordinate of player and closest enemy, choose action
		if closest_flag[0] > player_coordinate[0][0]:
			return "right"
		else:
			return "left"


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"E:\Reserch\GameAI\skier\image_processing_lab\sample2.png")
	action = rule_action(img)
	print(action)

