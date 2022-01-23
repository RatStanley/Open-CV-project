import cv2
import json
import click
import copy
from glob import glob

import numpy as np
from tqdm import tqdm

from typing import Dict


def empty_callback(value):
    pass



class Object:
    x_start = 0
    y_start = 0
    x_end = 0
    y_end = 0



img_org = cv2.imread("data/02.jpg", cv2.IMREAD_COLOR)
scale = 0.2
size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
img_org = cv2.resize(img_org, dsize=size_of_view)

img_show = img_org[:]
img_rgb = img_org[:]  # np.zeros(img_show.shape)

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

gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)

cv2.namedWindow('result')


key = ord('a')

ret, H_value = cv2.threshold(H_Orgin, 167, 255, cv2.THRESH_BINARY)
ret, S_value = cv2.threshold(S_Orgin, 104, 255, cv2.THRESH_BINARY)
ret, V_value = cv2.threshold(V_Orgin, 211, 255, cv2.THRESH_BINARY)

img_hsv_converted[:, :, 0] = H_value
img_hsv_converted[:, :, 1] = S_value
img_hsv_converted[:, :, 2] = V_value
# print(img_hsv_converted.shape)

gray_mask = cv2.bitwise_or(gray, H_value)
gray_mask = cv2.bitwise_or(gray_mask, S_value)
gray_mask = cv2.bitwise_or(gray_mask, V_value)
gray_mask[gray_mask != 255] = 0

rgb_mod = np.zeros(img_show.shape)
rgb_mod1 = cv2.bitwise_and(img_show[:, :, 0], gray_mask)
rgb_mod2 = cv2.bitwise_and(img_show[:, :, 1], gray_mask)
rgb_mod3 = cv2.bitwise_and(img_show[:, :, 2], gray_mask)


rgb_masked = np.dstack((rgb_mod1, rgb_mod2, rgb_mod3))
# x = 0
#
# while x != 1:
#     cv2.imshow('img_show', rgb_masked)
#     key = cv2.waitKey(30)
size_of_obj_y = 0
size_of_obj_x = 0

start_y = 0
end_y = 0
start_x = 0
end_x = 0

obj_list = [[]]
it = 0
# print(gray_mask.shape)
for x in range(gray_mask.shape[0]):
    for y in range( gray_mask.shape[1]):
        if gray_mask[x,y] == 255:
            if start_y != 0:
                start_y = y
            continue
        else:
            end_y = y
            if end_y - start_y > 2:
                obj_list.append([x,start_y,x,end_y])
                it = it + 1
            start_y = 0
            end_y = 0
    # print(x)
# print(it)
# print(obj_list[20000][0])
# print(obj_list[20000][1])
# print(obj_list[20000][2])
# print(obj_list[20000][3])
# print(obj_list)
# print(obj_list)
temp = 1
print(gray_mask[100,:])
for x_cord in range(gray_mask.shape[0]):
    for y_cord in range(gray_mask.shape[1]):
        if gray_mask[x_cord,y_cord] == 255:
            if temp == 1:
                gray_mask[x_cord, y_cord] = 99
            else:
                gray_mask[x_cord,y_cord] = temp
            temp = temp +1
        else:
            if temp > 1:
                gray_mask[x_cord, y_cord] = 200
            temp = 1
print(gray_mask[100,:])

# for x in range(1,600):
#     # print(x)
#     cv2.line(rgb_masked,(obj_list[x][0],obj_list[x][1]),(obj_list[x][2],obj_list[x][3]),(0, 255, 255),2)

while key != ord('q'):
    # Wait a little (30 ms) for a key press - this is required to refresh the image in our window

    cv2.imshow('result', rgb_masked)

    key = cv2.waitKey(30)
# TODO: Implement detection method.

apple = 0
banana = 0
orange = 0
print("test")



'''
cv2.namedWindow('img_show')

cv2.resizeWindow('result', 900, 900)

cv2.createTrackbar('HL', 'result', 0, 255, empty_callback)

cv2.createTrackbar('SL', 'result', 0, 255, empty_callback)

cv2.createTrackbar('VL', 'result', 0, 255, empty_callback)

cv2.createTrackbar('RL', 'result', 0, 255, empty_callback)

cv2.createTrackbar('GL', 'result', 0, 255, empty_callback)

cv2.createTrackbar('BL', 'result', 0, 255, empty_callback)

HL = cv2.getTrackbarPos('HL', 'result')
    SL = cv2.getTrackbarPos('SL', 'result')
    VL = cv2.getTrackbarPos('VL', 'result')
    RL = cv2.getTrackbarPos('RL', 'result')
    GL = cv2.getTrackbarPos('GL', 'result')
    BL = cv2.getTrackbarPos('BL', 'result')

    ret, H_value = cv2.threshold(H_Orgin, HL, 255, cv2.THRESH_BINARY)
    ret, S_value = cv2.threshold(S_Orgin, SL, 255, cv2.THRESH_BINARY)
    ret, V_value = cv2.threshold(V_Orgin, VL, 255, cv2.THRESH_BINARY)
    ret, R_value = cv2.threshold(rgb_masked[:, :, 2], RL, 255, cv2.THRESH_TOZERO)
    ret, G_value = cv2.threshold(rgb_masked[:, :, 1], GL, 255, cv2.THRESH_TOZERO)
    ret, B_value = cv2.threshold(rgb_masked[:, :, 0], BL, 255, cv2.THRESH_TOZERO)

    img_hsv_converted[:, :, 0] = H_value
    img_hsv_converted[:, :, 1] = S_value
    img_hsv_converted[:, :, 2] = V_value

    gray_mask = cv2.bitwise_or(gray, H_value)
    gray_mask = cv2.bitwise_or(gray_mask, S_value)
    gray_mask = cv2.bitwise_or(gray_mask, V_value)
    # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    # view =  cv2.cvtColor(img_hsv_converted, cv2.COLOR_RGB2GRAY)
    # mask = cv2.bitwise_not(gray_mask)
    gray_mask[gray_mask != 255] = 0

    img_rgb[:, :, 2] = R_value
    img_rgb[:, :, 1] = G_value
    img_rgb[:, :, 0] = B_value

    # img_rgb = cv2.bitwise_or(img_rgb, mask)

    img = np.hstack((img_org, img_hsv_converted))
    img = np.hstack((img, img_rgb))
    cv2.imshow('result', img)
    '''