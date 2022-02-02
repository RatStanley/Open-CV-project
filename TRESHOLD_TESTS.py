import cv2
import copy
import numpy as np


def empty_callback(value):
    pass

img_org = cv2.imread("data/00.jpg", cv2.IMREAD_COLOR)
scale = 0.2
size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
img_org = cv2.resize(img_org, dsize=size_of_view)

img_show = img_org[:]
img_rgb = img_org.copy()  # np.zeros(img_show.shape)

img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)

img_show = copy.deepcopy(img_org)
img_hsv_converted = copy.deepcopy(img_hsv)

img_rgb = img_show  # np.zeros(img_show.shape)
gray_mask = np.zeros(img_show.shape)
# img = np.concatenate((img_show, img_hsv, img_rgb), axis=1)
img = np.hstack((img_show, img_hsv))
img = np.hstack((img, img_rgb))

H_Orgin = img_hsv[:, :, 0]
S_Orgin = img_hsv[:, :, 1]
V_Orgin = img_hsv[:, :, 2]

R_Orgin = img_rgb[:, :, 0]
G_Orgin = img_rgb[:, :, 1]
B_Orgin = img_rgb[:, :, 2]

gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)

# cv2.namedWindow('result')
cv2.namedWindow('img_show')

# cv2.resizeWindow('result', 900, 900)

cv2.createTrackbar('HL', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('SL', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('VL', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('R', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('G', 'img_show', 0, 255, empty_callback)
cv2.createTrackbar('B', 'img_show', 0, 255, empty_callback)

key = ord('a')

ret, H_value = cv2.threshold(H_Orgin, 167, 255, cv2.THRESH_BINARY)
ret, S_value = cv2.threshold(S_Orgin, 104, 255, cv2.THRESH_BINARY)
ret, V_value = cv2.threshold(V_Orgin, 211, 255, cv2.THRESH_BINARY)

# img_hsv_converted[:, :, 0] = H_value
# img_hsv_converted[:, :, 1] = S_value
# img_hsv_converted[:, :, 2] = V_value
# # print(img_hsv_converted.shape)
#
#
# gray_mask = cv2.bitwise_or(gray, H_value)
# gray_mask = cv2.bitwise_or(gray_mask, S_value)
# gray_mask = cv2.bitwise_or(gray_mask, V_value)
# gray_mask[gray_mask != 255] = 0
#
rgb_mod = np.zeros(img_show.shape)
# rgb_mod1 = cv2.bitwise_and(img_show[:, :, 0], gray_mask)
# rgb_mod2 = cv2.bitwise_and(img_show[:, :, 1], gray_mask)
# rgb_mod3 = cv2.bitwise_and(img_show[:, :, 2], gray_mask)

# rgb_masked = np.dstack((rgb_mod1, rgb_mod2, rgb_mod3))
# x = 0
# print(rgb_masked.shape)
#
# print(gray_mask.shape)
#
# while x != 1:
#     cv2.imshow('img_show', rgb_masked)
#     key = cv2.waitKey(30)

while key != ord('q'):
    # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
    HL = cv2.getTrackbarPos('HL', 'img_show')
    SL = cv2.getTrackbarPos('SL', 'img_show')
    VL = cv2.getTrackbarPos('VL', 'img_show')
    R = cv2.getTrackbarPos('R', 'img_show')
    G = cv2.getTrackbarPos('G', 'img_show')
    B = cv2.getTrackbarPos('B', 'img_show')


    ret, H_value = cv2.threshold(H_Orgin, HL, 255, cv2.THRESH_BINARY)
    ret, S_value = cv2.threshold(S_Orgin, SL, 255, cv2.THRESH_BINARY)
    ret, V_value = cv2.threshold(V_Orgin, VL, 255, cv2.THRESH_BINARY)
    ret, R_value = cv2.threshold(R_Orgin, R, 255, cv2.THRESH_BINARY)
    ret, G_value = cv2.threshold(G_Orgin, G, 255, cv2.THRESH_BINARY)
    ret, B_value = cv2.threshold(B_Orgin, B, 255, cv2.THRESH_BINARY)


    # img_hsv_converted[:, :, 0] = H_value
    # img_hsv_converted[:, :, 1] = S_value
    # img_hsv_converted[:, :, 2] = V_value
    #
    # img_rgb[:, :, 0] = H_value
    # img_rgb[:, :, 1] = S_value
    # img_rgb[:, :, 2] = V_value

    gray_mask_hsv = cv2.bitwise_or(gray, H_value)
    gray_mask_hsv = cv2.bitwise_or(gray_mask_hsv, S_value)
    gray_mask_hsv = cv2.bitwise_or(gray_mask_hsv, V_value)
    gray_mask_rgb = cv2.bitwise_or(gray, R_value)
    gray_mask_rgb = cv2.bitwise_or(gray_mask_rgb, G_value)
    gray_mask_rgb = cv2.bitwise_or(gray_mask_rgb, B_value)

    gray_mask = cv2.bitwise_and(gray_mask_rgb, gray_mask_hsv)
    # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    view =  cv2.cvtColor(img_hsv_converted, cv2.COLOR_RGB2GRAY)
    # mask = cv2.bitwise_not(gray_mask)
    gray_mask[gray_mask != 255] = 0
    gray_mask = cv2.medianBlur(gray_mask, 31)


    # img_rgb = cv2.bitwise_or(img_rgb, mask)


    img = np.hstack((img, img_hsv_converted))
    # cv2.imshow('result', img_rgb)
    cv2.imshow('img_show', view)

    key = cv2.waitKey(30)
# TODO: Implement detection method.

apple = 0
banana = 0
orange = 0
print("test")