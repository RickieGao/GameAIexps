import glob as gb
import numpy as np
import cv2
img_names = gb.glob("*.npy")
# print(type(img_names))
# print(len(img_names))
# img_names.sort()
# print(img_names[1][-5])
# print(img_names[1][-6])
img = np.load(img_names[0])
# img = image_processing.PygameSurfaceToCV2Frame(img)
print(type(img))
cv2.imshow("img", img)
cv2.imwrite("ok.png", img)
cv2.waitKey(10000)
