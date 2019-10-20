import cv2
from random import choice
import imageProcessing

PLAYER_RADIUS = 57
ENEMY_RADIUS = 25
THRESH = 3


def rule_action(frame):
	"""
	function: according to the relative location of closest enemy and player return the action based rule
	:param frame: the input frame
	:return: the action based rule
	"""
	objects = imageProcessing.extract_objects(frame)
	player_coordinate = []
	enemy_coordinate = []
	for ob in objects:
		if PLAYER_RADIUS - THRESH < ob[1] < PLAYER_RADIUS + THRESH:
			player_coordinate.append(ob)
		if ENEMY_RADIUS - THRESH < ob[1] < ENEMY_RADIUS + THRESH:
			enemy_coordinate.append(ob)

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
		closest_enemy = max(enemy_coordinate, key=lambda x: x[0][1])

		# compare the X coordinate of player and closest enemy, choose action
		if closest_enemy[0][0] > player_coordinate[0][0][0]:
			return "right"
		else:
			return "left"


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"imageExtractionLab/sample7.png")
	action = rule_action(img)
	print(action)

