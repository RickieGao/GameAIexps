import cv2
from random import choice
from imageExtractionLab import imageProcessing

PLAYER_COLOR = [41, 212, 56]
BULLET_COLOR = [70, 213, 213]
ENEMY_COLOR = [61, 106, 212]
THRESH = 40


def rule_action(frame):
	"""
	function: according to the relative location of closest enemy and player return the action based rule
	:param frame: the input frame
	:return: the action based on rule
	"""
	player_coordinate = imageProcessing.extract_objects(frame, PLAYER_COLOR, THRESH)
	enemy_coordinate = imageProcessing.extract_objects(frame, ENEMY_COLOR, THRESH)

	# if there are more than one player, raise a error
	# if len(player_coordinate) > 1:
	# 	raise Exception("More than one player!")

	# action options
	options = ["left", "right"]

	# if no enemy or more than one player, return random choice
	if len(enemy_coordinate) == 0 or len(player_coordinate) > 1:
		return choice(options)
	else:
		# find the enemy that closest to player, record it at a tuple
		closest_enemy = max(enemy_coordinate, key=lambda x: x[1])

		# compare the X coordinate of player and closest enemy, choose action
		if closest_enemy[0] > player_coordinate[0][0]:
			return "right"
		else:
			return "left"


# test sample
if __name__ == '__main__':
	img = cv2.imread("ok.png")
	action = rule_action(img)
	print(action)

