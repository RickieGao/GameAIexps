import cv2
from random import choice
import imageProcessing
import numpy as np

PLAYER_COLOR = [1, 240, 255]
FLAG_COLOR = [1, 1, 255]
TREE_COLOR = [1, 255, 1]
THRESH = 40


def rule_action(frame):
	"""
	function: according to the relative location of closest tree and player return the action based rule
				The rule is to avoid the tree which is closest the player
	:param frame: the input frame
	:return: the action based rule
	"""
	# action options
	options = ["left", "right"]

	player_coordinate = imageProcessing.extract_objects(frame, PLAYER_COLOR, THRESH)
	tree_coordinate = imageProcessing.extract_objects(frame, TREE_COLOR, THRESH)

	# if no tree or no player or more than one player, return random choice
	if len(tree_coordinate) == 0 or len(player_coordinate) > 1 or len(player_coordinate) == 0:
		return choice(options)

	# distance between the trees and player
	distance = [((player_coordinate[0][0] - tree[0])**2 + (player_coordinate[0][1] - tree[1])**2)**0.5 for tree in tree_coordinate]

	# which tree is the closest tree
	min_distance_index = np.argmin(distance)

	# if the closest tree is in the right, go left
	if tree_coordinate[min_distance_index][0] > player_coordinate[0][0]:
		return "left"
	# if the closest tree is in the left, go right
	if tree_coordinate[min_distance_index][0] <= player_coordinate[0][0]:
		return "right"


# test sample
if __name__ == '__main__':
	img = cv2.imread(r"E:\Reserch\GameAI\skier\image_processing_lab\sample.png")
	action = rule_action(img)
	print(action)

