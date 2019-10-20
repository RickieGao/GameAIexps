import cv2
from random import choice
import imageProcessing

PLAYER_RADIUS = 57
ENEMY_RADIUS = 25
THRESH = 3
SAFE_HIGH = 480
PLAYER_HIGH = 700
SAFE_WIDTH = 125


def rule_action(frame):
	"""
	function: according to the relative location of enemy and player return the action based on rules
	:param frame: the input frame
	:return: the action based on rule
	"""
	objects = imageProcessing.extract_objects(frame)
	player_coordinate = []
	enemy_coordinate = []
	for ob in objects:
		if PLAYER_RADIUS - THRESH < ob[1] < PLAYER_RADIUS + THRESH:
			player_coordinate.append(ob)
		if ENEMY_RADIUS - THRESH < ob[1] < ENEMY_RADIUS + THRESH:
			enemy_coordinate.append(ob)

	# action options
	options = ["left", "right"]

	unsafe_zone = []
	for enemy in enemy_coordinate:
		if SAFE_HIGH < enemy[1] < PLAYER_HIGH and abs(enemy[0] - player_coordinate[0][0]) < SAFE_WIDTH:
			unsafe_zone.append(enemy)
	# if no enemy or more than one player, return random choice
	if len(enemy_coordinate) == 0 or len(player_coordinate) > 1:
		ruleAction = choice(options)
		safeAction = "safe"
	elif len(unsafe_zone) == 0:  # there are enemy in the screen, but no enemy in the unsafe zone.
		# find the enemy that closest to player, record it at a tuple
		closest_enemy = min(enemy_coordinate, key=lambda x: abs(x[0][0] - player_coordinate[0][0][0]))
		# compare the X coordinate of player and closest enemy, choose action
		if closest_enemy[0][0] > player_coordinate[0][0][0]:
			ruleAction = "right"
			safeAction = "safe"
		else:
			ruleAction = "left"
			safeAction = "safe"
	elif len(unsafe_zone) == 1:  # there is one enemy in the unsafe zone
		ruleAction = choice(options)
		if unsafe_zone[0][0][0] > player_coordinate[0][0][0]:
			safeAction = "left"
		else:
			safeAction = "right"
	else:
		ruleAction = choice(options)
		most_dangerous_enemy = max(unsafe_zone, key=lambda x: x[0][1])
		if most_dangerous_enemy[0][0] > player_coordinate[0][0][0]:
			safeAction = "left"
		else:
			safeAction = "right"
	return ruleAction, safeAction


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"imageExtractionLab/sample6.png")
	action = rule_action(img)
	print(action)
