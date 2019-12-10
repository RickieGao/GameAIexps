import glob as gb
import cv2
img_names = gb.glob("image_data//rule_input_image//*.npy")
print(type(img_names))
print(len(img_names))
img_names.sort()
print(img_names[1][-5])
print(img_names[1][-6])
# for i in range(4):
# 	img = cv2.imread(img_names[i])
# 	cv2.imshow('img', img)
# 	cv2.waitKey(1000)
# img = cv2.imread(img_names[0])
# cv2.imshow('img', img)
# cv2.waitKey(10000)
