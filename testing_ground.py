import cv2
import copy
import numpy as np

img_org = cv2.imread("data/03.jpg", cv2.IMREAD_COLOR)
scale = 0.2
size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
img_org = cv2.resize(img_org, dsize=size_of_view)
img_show = img_org[:]
img_rgb = img_org[:]
img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
img_hsv_converted = copy.deepcopy(img_hsv)
gray_mask = np.zeros(img_show.shape)
img = np.hstack((img_show, img_hsv))
img = np.hstack((img, img_rgb))
H_Orgin = img_hsv[:, :, 0]
S_Orgin = img_hsv[:, :, 1]
V_Orgin = img_hsv[:, :, 2]
gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('result')
key = ord('a')
ret, H_value = cv2.threshold(H_Orgin, 220, 255, cv2.THRESH_BINARY)
ret, S_value = cv2.threshold(S_Orgin, 104, 255, cv2.THRESH_BINARY)  # 104
ret, V_value = cv2.threshold(V_Orgin, 222, 255, cv2.THRESH_BINARY)
img_hsv_converted[:, :, 0] = H_value
img_hsv_converted[:, :, 1] = S_value
img_hsv_converted[:, :, 2] = V_value
gray_mask = cv2.bitwise_or(gray, H_value)
gray_mask = cv2.bitwise_or(gray_mask, S_value)
gray_mask = cv2.bitwise_or(gray_mask, V_value)
gray_mask[gray_mask != 255] = 0
gray_mask = cv2.medianBlur(gray_mask, 31)
rgb_mod = np.zeros(img_show.shape)
rgb_mod1 = cv2.bitwise_and(img_show[:, :, 0], gray_mask)
rgb_mod2 = cv2.bitwise_and(img_show[:, :, 1], gray_mask)
rgb_mod3 = cv2.bitwise_and(img_show[:, :, 2], gray_mask)
rgb_masked = np.dstack((rgb_mod1, rgb_mod2, rgb_mod3))


#######################################################################################################################

# def not_end_of_object(img, x,y):
#     if img[x,y] == 255:
#         return True
#     return False


# def search_for_obj_size(img, x_s, y_s):
#     x_1 = x_s
#     y_1 = y_s
#     x_2 = x_s
#     y_2 = y_s
#     x_3 = x_s
#     y_3 = y_s
#     dif_1 = -2
#     dif_2 = -2
#     dif_3 = 2
#     x_list = []
#     y_list = []
#     hited = 0
#     while True:
#         if x_1 + dif_1 > img.shape[0]-2:
#             dif_1 = dif_1 * -1
#             x_1 = img.shape[0]-2
#         elif x_2 + dif_2  > img.shape[0]and y_2 + dif_2> img.shape[1]-2:
#             dif_2 = dif_2 * -1
#             x_2 = img.shape[0] - 2
#             y_2 = img.shape[1] - 2
#         elif x_3 + dif_3  > img.shape[0]and y_3 + dif_3> img.shape[1]-2:
#             dif_3 = dif_3 * -1
#             x_3 = img.shape[0] - 2
#             y_3 = img.shape[1] - 2
#
#
#         if img[x_1 + dif_1, y_1] == 255:
#             if dif_1 == 0:
#                 x_1 += dif_1
#             else:
#                 x_1 += dif_1
#                 y_1 += dif_1
#         else:
#             dif_1 = dif_1*-1
#             x_list.append(x_1)
#             y_list.append(y_1)
#             hited += 1
#
#         if img[x_2 + dif_2, y_2 + dif_2] == 255:
#             x_1 += dif_2
#             y_2 += dif_2
#         else:
#             dif_2 = dif_2 * -1
#             x_list.append(x_2)
#             y_list.append(y_2)
#             hited += 1
#
#         if img[x_3 - dif_3, y_3 - dif_3] == 255:
#             x_3 += dif_3
#             y_3 += dif_3
#
#         else:
#             dif_3 = dif_3 * -1
#             x_list.append(x_3)
#             y_list.append(y_3)
#             hited += 1
#         if hited > 100:
#             return [x_list, y_list]

# for x_cord in range(gray_mask.shape[0]):
#     for y_cord in range(gray_mask.shape[1]):
#         if gray_mask[x_cord,y_cord] == 255:
#             [x_list, y_list] = search_for_obj_size(gray_mask, x_cord, y_cord)
#             # x_min = min(x_list)
#             # x_max = max(x_list)
#             # y_min = min(y_list)
#             # y_max = max(y_list)
#             # if x_max - x_min > 100 and y_max - y_min > 100:
#             #     list.append([x_min, x_max,y_min,y_max])
#             print(x_list)
#             print(y_list)
#             # break
#     # if not len(list) == 0:
#     #     break


# def search_for_obj_size(img, x_s, y_s):
#
#     for x in range(x_s+1,img.shape[0]):
#         if img[x,y_s] == 255:
#             continue
#         else:
#             return [x_s,x-1]
#
#
#
# x_list = []
# y_list = []
# list = []
# for x_cord in range(gray_mask.shape[0]):
#     for y_cord in range(gray_mask.shape[1]):
#         if gray_mask[x_cord,y_cord] == 255:
#             [x_min, x_max] = search_for_obj_size(gray_mask, x_cord, y_cord)
#             if x_max - x_min > 50:
#                 x_list.append([x_min,x_max])
#                 break
#             # break
#     y_list.append([y_cord,x_list])
#     # if not len(list) == 0:
#     #     break
#
# print((y_list[0]))


# print(list)
# x_m = min(x_list)
# x_mx = max(x_list)
# y_m = min(y_list)
# y_mx = max(y_list)
# rgb_masked = cv2.rectangle(rgb_masked,(y_mx,x_m),(y_m,x_mx),(255,0,0),2)

# for pkt in list:
#     rgb_masked = cv2.rectangle(rgb_masked,(pkt[2],pkt[0]),(pkt[3],pkt[1]),(255,0,0),2)


while key != ord('q'):
    # cv2.imshow('result', gray_mask)
    cv2.imshow('result', rgb_masked)
    key = cv2.waitKey(30)
# TODO: Implement detection method.
