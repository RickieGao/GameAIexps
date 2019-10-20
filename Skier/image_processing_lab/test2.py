import cv2
from random import choice
import imageProcessing
import numpy as np

PLAYER_COLOR = [1, 240, 255]
FLAG_COLOR = [1, 1, 255]
TREE_COLOR = [1, 255, 1]
THRESH = 40

frame = cv2.imread(r"E:\Reserch\GameAI\skier\image_processing_lab\sample1.png")

player_coordinate = imageProcessing.extract_objects(frame, PLAYER_COLOR, THRESH)
tree_coordinate = imageProcessing.extract_objects(frame, TREE_COLOR, THRESH)

distance = [((player_coordinate[0][0] - tree[0])**2 + (player_coordinate[0][1] - tree[1])**2)**0.5 for tree in tree_coordinate]

print(tree_coordinate)
print(player_coordinate)
min_distance_index = np.argmin(distance)
print(tree_coordinate[min_distance_index])
