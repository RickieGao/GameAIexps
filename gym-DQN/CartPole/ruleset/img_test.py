import numpy as np
import glob as gb
import cv2

POLE_COLOR = [102, 153, 204]
THRESH = 40

img_names = gb.glob("npy_dates//*.npy")
img_names.sort()


def state_preprocess(state_tensor):
    state = state_tensor.squeeze(0)
    fetch_state = state[3, :, :]
    fetch_state = fetch_state.unsqueeze(0)
    fetch_state = fetch_state.repeat([4, 1, 1])
    state_tensor = np.array(fetch_state)
    state_tensor = state_tensor.transpose((1, 2, 0))
    frame = cv2.cvtColor(state_tensor, cv2.COLOR_RGB2GRAY)
    return frame


def extract_objects(frame, bgr_list, thresh):
    """
	function: extract the object specified color
	:param frame: image to be processing
	:param bgr_list: BGR color of objects to be extracted
	:param thresh: range of color
	:return: a list of coordinates of the center of extracted objects, just like [()]
	"""
    minBGR = np.array([bgr_list[0] - thresh, bgr_list[1] - thresh, bgr_list[2] - thresh])
    maxBGR = np.array([bgr_list[0] + thresh, bgr_list[1] + thresh, bgr_list[2] + thresh])

    mask = cv2.inRange(frame, minBGR, maxBGR)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # visualization, draw a bounding box around the detected barcode and display the image
    # cv2.drawContours(closing, [box], -1, (255, 255, 255), 3)
    # cv2.imshow("Image", closing)
    # cv2.waitKey(0)

    return box


def rule_action(frame):
    """
	function: according to the incline orientation of the pole return the action based rule
	:param frame: the input frame
	:return: the action based rule
	"""
    frame = state_preprocess(frame)
    box = extract_objects(frame, POLE_COLOR, THRESH)

    # numpy array to list
    box_list = box.tolist()
    # sorted according to Y coordinate
    box_list = sorted(box_list, key=lambda x: x[1])
    top = box_list[: 2]
    bottom = box_list[2:]
    top = sorted(top, key=lambda x: x[0])
    bottom = sorted(bottom, key=lambda x: x[0])
    if top[0][0] >= bottom[0][0]:
        return "right"
    else:
        return "left"


# test sample
if __name__ == '__main__':
    # for img_name in img_names:
    #     imgnpy_file = np.load(img_name)
    #     # action = rule_action(imgnpy_file)
    #     # print(action)
    #     print(type(imgnpy_file), ':')
    imgnpy1 = img_names[14]
    imgnpy_file = np.load(imgnpy1)
    print(type(imgnpy_file), ':')
    print(imgnpy_file)
    print(imgnpy_file[2])
    # print(np.shape(imgnpy_file))
    print('finished!')
