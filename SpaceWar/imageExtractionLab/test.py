import cv2
import numpy as np

"""
rgb value:
submarine: [53, 187, 187]
human: [200, 72, 66]
enemy: [92, 186,92]

PLAYER_COLOR = [41, 212, 56]
BULLET_COLOR = [70, 213, 213]
ENEMY_COLOR = [61, 106, 212]
"""


def CropFrame(frame):
	return frame[35:160, 8:165]


def ExtractSubmarine(frame):
	# BGR color of submarine
	submarine_BGRcolor = [41, 212, 56]
	thresh = 40

	minBGR = np.array([submarine_BGRcolor[0] - thresh, submarine_BGRcolor[1] - thresh, submarine_BGRcolor[2] - thresh])
	maxBGR = np.array([submarine_BGRcolor[0] + thresh, submarine_BGRcolor[1] + thresh, submarine_BGRcolor[2] + thresh])

	mask = cv2.inRange(frame, minBGR, maxBGR)
	cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	rets = []
	for cnt in cnts:
		# cnt is a numpy array of coordinates
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		rets.append(center)
	# for cnt in cnts:
	# 	((x, y), radius) = cv2.minEnclosingCircle(cnt)
	# 	x = int(x)
	# 	y = int(y)
	# 	center = (x, y)
	# 	ret = [center]
	# 	for vertex in cnt:
	# 		# vertex is [[x, y]]
	# 		ret.append(tuple(vertex[0]))
	# 	rets.append(ret)

	cv2.imshow("Result BGR", mask)

	print(len(cnts))
	print(rets)
	cv2.waitKey()


if __name__=='__main__':
	im = cv2.imread("sample222.png")
	# croped = CropFrame(im)
	ExtractSubmarine(im)


