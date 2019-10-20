import imageio
import cv2
im = imageio.imread('sample.png')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', im_gray)
cv2.waitKey(0)
retval, im_at_fixed = cv2.threshold(im_gray, 175, 255, cv2.THRESH_BINARY)	# pixel greater than second-arg will turn white(255)
cv2.imshow('fixed', im_at_fixed)
cv2.waitKey(0)