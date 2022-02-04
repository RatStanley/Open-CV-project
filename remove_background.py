import cv2
import copy
import numpy as np


def empty_callback(value):
    pass


cv2.namedWindow('img_show')
key = ord('a')


img_org = cv2.imread("data/00.jpg", cv2.IMREAD_COLOR)
scale = 0.2
size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
img_org = cv2.resize(img_org, dsize=size_of_view)

img_show = img_org[:]
img_rgb = img_org.copy()  # np.zeros(img_show.shape)

img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
img_hsv = cv2.medianBlur(img_hsv, 41)
img_show = copy.deepcopy(img_org)
img_hsv_converted = copy.deepcopy(img_hsv)

img_rgb = img_show  # np.zeros(img_show.shape)
gray_mask = np.zeros(img_show.shape)
# img = np.concatenate((img_show, img_hsv, img_rgb), axis=1)


gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)

# cv2.namedWindow('result')




cv2.createTrackbar('HL', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('SL', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('VL', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('HH', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('SH', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('VH', 'img_show', 0, 255, empty_callback)






while key != ord('q'):
    # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
    HL = cv2.getTrackbarPos('HL', 'img_show')
    SL = cv2.getTrackbarPos('SL', 'img_show')
    VL = cv2.getTrackbarPos('VL', 'img_show')
    HH = cv2.getTrackbarPos('HH', 'img_show')
    SH = cv2.getTrackbarPos('SH', 'img_show')
    VH = cv2.getTrackbarPos('VH', 'img_show')

    low_green = np.array([HL, SL, VL])
    high_green = np.array([HH, SH, VH])

    mask = cv2.inRange(img_hsv, low_green, high_green)

    # mask = 255 - mask
    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    cv2.imshow('img_show', res)

    key = cv2.waitKey(30)
# TODO: Implement detection method.

apple = 0
banana = 0
orange = 0
print("test")