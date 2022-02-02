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

def image_to_contur(H,S,V,path = " ",img_org_ = None):
    if not path == " ":
        img = cv2.imread(path)
    else:
        img = img_org_.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    H_Orgin = img[:, :, 0]
    S_Orgin = img[:, :, 1]
    V_Orgin = img[:, :, 2]
    ret, H_value = cv2.threshold(H_Orgin, H, 255, cv2.THRESH_BINARY) #220
    ret, S_value = cv2.threshold(S_Orgin, S, 255, cv2.THRESH_BINARY) #100 # 104
    ret, V_value = cv2.threshold(V_Orgin, V, 255, cv2.THRESH_BINARY) #202
    img[:, :, 0] = H_value
    img[:, :, 1] = S_value
    img[:, :, 2] = V_value

    img = cv2.medianBlur(img, 31)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[img != 0] = 255

    contur_list, heirachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contur_list))
    return contur_list, heirachy

def contur_compare(counturs_background_, counturs_obj,size):
    dist_list = []
    for cnt in counturs_background_:
        retval = cv2.matchShapes(cnt, counturs_obj, 1, 0)
        print(retval)
        dist_list.append(retval)

    sorted_list = dist_list.copy()
    sorted_list.sort()  # sorts the list from smallest to largest
    obj_count = []
    rest = []
    for index in range(len(sorted_list)):
        if sorted_list[index] < size:
            ind = dist_list.index(sorted_list[index])
            obj_count.append(counturs_background_[ind])
        else:
            ind = dist_list.index(sorted_list[index])
            rest.append(counturs_background_[ind])
    return obj_count,rest

#maski obiektów
org = cv2.imread("data/00.JPG")
scale = 0.2
size_of_view = (int(org.shape[1] * scale), int(org.shape[0] * scale))
org = cv2.resize(org, dsize=size_of_view)

contours_background, heirachy_bacground = image_to_contur(img_org_=org,H=255,S=80,V=215)#H=200,S=100,V=202
ref_banana, heirachy_banana = image_to_contur(path="masks/BANAN.JPG",H=200,S=100,V=202)


ref_orange, heirachy_orange = image_to_contur(path="masks/JABLKO_1.JPG",H=255,S=93,V=207)

reference_contour_banana = ref_banana[0]
reference_contour_orange = ref_orange[0]

print(len(contours_background))
banana_cnts,rest = contur_compare(contours_background,reference_contour_banana,1)
print(len(rest))
orange_cnts,rest = contur_compare(rest,reference_contour_orange,0.3)
print(len(rest))


with_contours = cv2.drawContours(org,banana_cnts,-1,(255,0,0),3)
with_contours = cv2.drawContours(org,orange_cnts,-1,(0,255,0),3)


# with_contours = cv2.drawContours(org,rest,-1,(255,255,0),3)

while key != ord('q'):
    # cv2.imshow('result', gray_mask)
    cv2.imshow('result', with_contours)
    key = cv2.waitKey(30)
# TODO: Implement detection method.
