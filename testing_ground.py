import cv2
import copy
import numpy as np

# img_org = cv2.imread("data/03.jpg", cv2.IMREAD_COLOR)
# scale = 0.2
# size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
# img_org = cv2.resize(img_org, dsize=size_of_view)

cv2.namedWindow('result')
key = ord('a')


#######################################################################################################################

def image_to_contur(path = " ", img_org_ = None):
    if not path == " ":
        img = cv2.imread(path)
    else:
        img = img_org_.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    H_Orgin = img[:, :, 0]
    S_Orgin = img[:, :, 1]
    V_Orgin = img[:, :, 2]
    ret, H_value = cv2.threshold(H_Orgin, 200, 255, cv2.THRESH_BINARY) #220
    ret, S_value = cv2.threshold(S_Orgin, 104, 255, cv2.THRESH_BINARY) #100 # 104
    ret, V_value = cv2.threshold(V_Orgin, 210, 255, cv2.THRESH_BINARY) #202
    img[:, :, 0] = H_value
    img[:, :, 1] = S_value
    img[:, :, 2] = V_value

    img = cv2.medianBlur(img, 31)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[img != 0] = 255

    contur_list, heirachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contur_list))
    return contur_list, heirachy


org = cv2.imread("data/09.JPG")
scale = 0.2
size_of_view = (int(org.shape[1] * scale), int(org.shape[0] * scale))
org = cv2.resize(org, dsize=size_of_view)

contours_background, heirachy_bacground = image_to_contur(img_org_=org)
ref_banana, heirachy_banana = image_to_contur("masks/BANAN.JPG")

reference_contour = ref_banana[0]
dist_list= []

for cnt in contours_background:
    retval= cv2.matchShapes(cnt, reference_contour,1,0)
    dist_list.append(retval)

sorted_list= dist_list.copy()
sorted_list.sort() # sorts the list from smallest to largest
banana_cnts= []
for index in range(len(sorted_list)):
    if sorted_list[index] < 1:
        ind = dist_list.index(sorted_list[index])
        banana_cnts.append(contours_background[ind])
    else:
        break


# ind1_dist= dist_list.index(sorted_list[0])
# ind2_dist= dist_list.index(sorted_list[1])
# print(len(dist_list))

# banana_cnts.append(contours_background[ind1_dist])
# banana_cnts.append(contours_background[ind2_dist])

with_contours = cv2.drawContours(org,banana_cnts,-1,(255,0,0),3)

while key != ord('q'):
    # cv2.imshow('result', gray_mask)
    cv2.imshow('result', with_contours)
    key = cv2.waitKey(30)
# TODO: Implement detection method.
