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

def look_for_any_else(list,var):
    for x in range(10):
        if len(list) > (var+10):
            if list[var][0]+1 == list[var+x+1]:
                return [True,x]
            else:
                break
    return [False, 1]


def look_for_gap(list, var, start_x, max_value, min_value,repetition_list,size_min):

    offset_fun = look_for_any_else(list,var)
    offset = offset_fun[1]
    if not var > len(list[:]) - 2:
        if  2 > list[var+1][0] - list[var][0]:
            if var > len(list[:]) - 2:
                return [start_x, list[var][0], min_value, max_value]
            range_a = range(list[var][1]-size_min, list[var][2]+size_min)
            range_b = range(list[var + offset][1]-size_min, list[var + offset][2]+size_min)

            if ((list[var][1] in range_b) or (list[var][2] in range_b) or (list[var + offset][1] in range_a) or (
                    list[var + offset][2] in range_a)):
                if list[var][1] < min_value:
                    min_value = list[var][1]
                if list[var][2] > max_value:
                    max_value = list[var][2]
                return look_for_gap(list, var + 1+offset, start_x, max_value, min_value,repetition_list,size_min)
            else:
                return [start_x, list[var][0], min_value, max_value,var]
        else:
            return [start_x, list[var][0], min_value, max_value,var]
    else:
        return [start_x, list[var-1][0], min_value, max_value, var]


img_org = cv2.imread("data/03.jpg", cv2.IMREAD_COLOR)
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

# ret, H_value = cv2.threshold(H_Orgin, 167, 255, cv2.THRESH_BINARY)
# ret, S_value = cv2.threshold(S_Orgin, 104, 255, cv2.THRESH_BINARY)
# ret, V_value = cv2.threshold(V_Orgin, 211, 255, cv2.THRESH_BINARY)

ret, H_value = cv2.threshold(H_Orgin, 220, 255, cv2.THRESH_BINARY)
ret, S_value = cv2.threshold(S_Orgin, 104, 255, cv2.THRESH_BINARY)
ret, V_value = cv2.threshold(V_Orgin, 222, 255, cv2.THRESH_BINARY)


img_hsv_converted[:, :, 0] = H_value
img_hsv_converted[:, :, 1] = S_value
img_hsv_converted[:, :, 2] = V_value
# print(img_hsv_converted.shape)

gray_mask = cv2.bitwise_or(gray, H_value)
gray_mask = cv2.bitwise_or(gray_mask, S_value)
gray_mask = cv2.bitwise_or(gray_mask, V_value)
gray_mask[gray_mask != 255] = 0

gray_mask = cv2.medianBlur(gray_mask, 5)

rgb_mod = np.zeros(img_show.shape)
rgb_mod1 = cv2.bitwise_and(img_show[:, :, 0], gray_mask)
rgb_mod2 = cv2.bitwise_and(img_show[:, :, 1], gray_mask)
rgb_mod3 = cv2.bitwise_and(img_show[:, :, 2], gray_mask)

rgb_masked = np.dstack((rgb_mod1, rgb_mod2, rgb_mod3))
#
temp = 1
start_y = 0
end_y = 0
list_y = []
pr = False
for x_cord in range(gray_mask.shape[0]):
    for y_cord in range(gray_mask.shape[1]):
        if gray_mask[x_cord, y_cord] == 255:
            if temp == 1:
                start_y = y_cord
            temp = temp + 1
        else:
            if temp > 10:
                end_y = y_cord - 1
                list_y.append([x_cord, start_y, end_y])
            temp = 1
    temp = 1

list_x = []
end_x = 0
temp = 1
temp = 1
for y_cord in range(gray_mask.shape[1]):
    for x_cord in range(gray_mask.shape[0]):
        if gray_mask[x_cord, y_cord] == 255:
            if temp == 1:
                start_x = x_cord
            temp = temp + 1
        else:
            if temp > 10:
                end_x = x_cord - 1
                list_x.append([y_cord, start_x, end_x])
            temp = 1
    temp = 1

obj = []
# test  = look_for_gap(list_y,10,list_y[10][0],0,1000)
list_collected_x = []
ind = 1
for i in range(len(list_x[:])-1):
    if list_x[i][0] == list_x[i+1][0]:
        ind += 1
    else:
        list_collected_x.append([ind,i - ind+1, list_x[i][0]])
        ind = 1
print((list_collected_x))
print(len(list_x))
# sum = 0
# for x,y,z in list_collected_x:
#     sum += x
# print(sum)
chce_koniec = 0
while True:

    for size in range(len(list_y[:])):
        temp_var = look_for_gap(list_y, size, list_y[size][0], 0, 1000,list_collected_x,50)

        if temp_var[1] - temp_var[0] > 50 and temp_var[3] - temp_var[2] > 50:
            obj.append(temp_var)
            del list_y[:obj[-1][4]]
            break

    chce_koniec +=1
    if chce_koniec > len(list_y[:]):
        break
#
# while True:
#
#     for size in range(len(list_x[:])):
#         temp_var = look_for_gap(list_x, size, list_x[size][0], 0, 1000,list_collected_x,50)
#
#         if temp_var[1] - temp_var[0] > 50 and temp_var[3] - temp_var[2] > 50:
#             obj.append(temp_var)
#             del list_x[:obj[-1][4]]
#             break
#
#     chce_koniec +=1
#     if chce_koniec > len(list_x[:]):
#         break

print((obj[:]))

# for cord in range(list_y):
#     if
# print(list_x)
# print(list_y)


# list_collected_y = []
# ind = 1
# for i in range(len(list_x[:])-1):
#     if list_x[i][0] == list_x[i+1][0]:
#         ind += 1
#     else:
#         list_collected_x.append([ind,i - ind+1, list_x[i][0]])
#         ind = 1


# print(list)
# print(img_org.shape)
#
#
#
# print(list)
for x in range(len(list_y[:])):
    cv2.line(rgb_masked, (list_y[x][1], list_y[x][0]), (list_y[x][2], list_y[x][0]), (0, 0, 255), 1)
#
# for x in range(len(list_x[:])):
#     cv2.line(rgb_masked, (list_x[x][0], list_x[x][1]), (list_x[x][0], list_x[x][2]), (255, 0, 0), 1)

# cv2.line(rgb_masked, (list[0][1], list[0][0]), (list[0][2], list[0][0]), (255, 0, 0), 1)

for i in range(len(obj[:])):
    rgb_masked = cv2.rectangle(rgb_masked, (obj[i][3], obj[i][1]), (obj[i][2], obj[i][0]), (0, 255, 0), 2) # dla y

# for i in range(len(obj[:])):
#     rgb_masked = cv2.rectangle(rgb_masked, (obj[i][0], obj[i][2]), (obj[i][1], obj[i][3]), (0, 255, 0), 2)

while key != ord('q'):
    # Wait a little (30 ms) for a key press - this is required to refresh the image in our window

    cv2.imshow('result', rgb_masked)

    key = cv2.waitKey(30)
# TODO: Implement detection method.

apple = 0
banana = 0
orange = 0
print("test")

# min_value = 100;
# max_value = 0;
# ind = 1
# # and ((list[i][1] < list[i+1][1]) or (list[i+1][1] < list[i+1][2]))
# # range_a = range(list[i][1], list[i][2])
# # range_b = range(list[i][1], list[i][2])
# for i in range(len(list[:])-1):
#     if list[i][0] == list[i+1][0]:
#         ind += 1
#     else:
#         list_in_range.append([ind,i - ind+1, list[i][0]])
#         ind = 1
#
# print(list_in_range[4])
# print(list)
# # print(list[list[:][0] > list[:+1][0]])
# obj = []
# for r in range(10):     #range(len(list_in_range[:])):
#     for l in range(list_in_range[r][0]):
#         range_a = range(list[r+i][1], list[r+i][2])
#         range_b = range(list[r+i][1], list[r+i][2])
#         print(list[r+l])


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
