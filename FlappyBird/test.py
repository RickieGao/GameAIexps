import glob as gb
import numpy as np
from ruleset import image_processing
import cv2
img_names = gb.glob("image_data//rule_image//*.npy")
# print(type(img_names))
# print(len(img_names))
# img_names.sort()
# print(img_names[1][-5])
# print(img_names[1][-6])
img = np.load(img_names[1])
# img = image_processing.PygameSurfaceToCV2Frame(img)
print(type(img))
cv2.imshow("img", img)
cv2.waitKey(10000)
