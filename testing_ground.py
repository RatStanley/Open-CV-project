import cv2
import copy
import numpy as np

img_org = cv2.imread("data/03.jpg", cv2.IMREAD_COLOR)
scale = 0.2
size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
img_org = cv2.resize(img_org, dsize=size_of_view)

cv2.namedWindow('result')
key = ord('a')
#######################################################################################################################

def image_to_contur(path):
    img = cv2.imread("masks/BANAN.JPG")
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    H_Orgin = img[:, :, 0]
    S_Orgin = img[:, :, 1]
    V_Orgin = img[:, :, 2]
    ret, H_value = cv2.threshold(H_Orgin, 220, 255, cv2.THRESH_BINARY)
    ret, S_value = cv2.threshold(S_Orgin, 100, 255, cv2.THRESH_BINARY)  # 104
    ret, V_value = cv2.threshold(V_Orgin, 202, 255, cv2.THRESH_BINARY)
    img[:, :, 0] = H_value
    img[:, :, 1] = S_value
    img[:, :, 2] = V_value

    img = cv2.medianBlur(img, 31)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[img != 0] = 255

    contur_list, heirachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contur_list))
    return contur_list,heirachy



# banana = cv2.imread("masks/BANAN.JPG")
# banana_mask = banana.copy()
# banana_mask = cv2.cvtColor(banana_mask, cv2.COLOR_BGR2HSV)
#
# H_Orgin = banana_mask[:, :, 0]
# S_Orgin = banana_mask[:, :, 1]
# V_Orgin = banana_mask[:, :, 2]
# ret, H_value = cv2.threshold(H_Orgin, 220, 255, cv2.THRESH_BINARY)
# ret, S_value = cv2.threshold(S_Orgin, 100, 255, cv2.THRESH_BINARY)  # 104
# ret, V_value = cv2.threshold(V_Orgin, 202, 255, cv2.THRESH_BINARY)
# banana_mask[:, :, 0] = H_value
# banana_mask[:, :, 1] = S_value
# banana_mask[:, :, 2] = V_value
#
# banana_mask = cv2.medianBlur(banana_mask, 31)
#
# banana_mask = cv2.cvtColor(banana_mask,cv2.COLOR_RGB2GRAY)
# banana_mask[banana_mask != 0] = 255
#
# contur_list, heirachy = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)

# H_Orgin = img_org[:, :, 0]
# S_Orgin = img_org[:, :, 1]
# V_Orgin = img_org[:, :, 2]
# ret, H_value = cv2.threshold(H_Orgin, 220, 255, cv2.THRESH_BINARY)
# ret, S_value = cv2.threshold(S_Orgin, 100, 255, cv2.THRESH_BINARY)  # 104
# ret, V_value = cv2.threshold(V_Orgin, 202, 255, cv2.THRESH_BINARY)
# img_org[:, :, 0] = H_value
# img_org[:, :, 1] = S_value
# img_org[:, :, 2] = V_value
# img_org = cv2.medianBlur(img_org, 31)
#
# img_org = cv2.cvtColor(img_org,cv2.COLOR_RGB2GRAY)
# img_org[img_org != 0] = 255
#
# contur_list, heirachy = cv2.findContours(img_org, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# conturs = cv2.drawContours(img_org,contur_list,-1,(255,0,0),5)
#
contur_list, heirachy = image_to_contur("masks/BANAN.JPG")
print(len(contur_list))
#to tylko by pokazać że działa
banana = cv2.imread("masks/BANAN.JPG")
conturs = cv2.drawContours(banana,contur_list,-1,(255,0,0),5)
#szukanie konturu

# contur_list2, heirachy2 = image_to_contur("masks/BANAN.JPG")

while key != ord('q'):
    # cv2.imshow('result', gray_mask)
    cv2.imshow('result', conturs)
    key = cv2.waitKey(30)
# TODO: Implement detection method.
