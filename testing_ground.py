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

    return contur_list, heirachy


def contur_compare(counturs_background_, counturs_obj,size):
    dist_list = []
    for cnt in counturs_background_:
        retval = cv2.matchShapes(cnt, counturs_obj, 1, 0)
        # print(retval)
        dist_list.append(retval)

    # sorted_list = dist_list.copy()
    # sorted_list.sort()  # sorts the list from smallest to largest
    obj_count = []
    rest = []
    for index in range(len(dist_list)):
        if dist_list[index] < size:
            ind = dist_list.index(dist_list[index])
            print(len(counturs_background_[ind]))
            if len(counturs_background_[ind]) > 100:
                obj_count.append(counturs_background_[ind])
        else:
            ind = dist_list.index(dist_list[index])
            rest.append(counturs_background_[ind])
    return obj_count,rest

#maski obiektów
apple = 0
banana = 0
orange = 0
org = cv2.imread("data/05.JPG")

scale = 0.2
size_of_view = (int(org.shape[1] * scale), int(org.shape[0] * scale))
org = cv2.resize(org, dsize=size_of_view)

contours_background_banana, heirachy_bacground = image_to_contur(img_org_=org, H=200, S=100, V=215)#H=200,S=100,V=202
ref_banana, heirachy_banana = image_to_contur(path="masks/BANAN.JPG",H=200,S=102,V=210)

contours_background_orange, heirachy_bacground_orange = image_to_contur(img_org_=org, H=255, S=210, V=235)#H=255, S=220, V=255

ref_orange, heirachy_orange = image_to_contur(path="masks/POMA.JPG",H=220,S=142,V=239)#,H=255,S=210,V=255

reference_contour_banana = ref_banana[0]
reference_contour_orange = ref_orange[0]

banana_cnts,rest_1 = contur_compare(contours_background_banana, reference_contour_banana, 1)
orange_cnts,rest_orange = contur_compare(contours_background_orange,reference_contour_orange,0.3)

np.save('file_name.txt', ref_banana)


banana = len(banana_cnts)
orange = len(orange_cnts)
print(len(orange_cnts))

with_contours = cv2.drawContours(org,banana_cnts,-1,(255,0,0),3)
with_contours = cv2.drawContours(org,orange_cnts,-1,(0,255,255),3)

orange_rect = []
for rest in orange_cnts:
    x,y,w,h = cv2.boundingRect(rest)
    orange_rect.append([x,y,w,h])
    cv2.rectangle(with_contours,(x,y),(x+w,y+h),(255,255,0),2)


for rest in rest_1:
    # for cord in orange_rect:


    x, y, w, h = cv2.boundingRect(rest)
    if len(rest) > 100:
        # print(len(rest))
        if len(orange_cnts) > 0:
            is_in_oragne = False
            for cord in orange_rect:
                px1, py1, px2, py2 = x, y, x+w, y+h
                hx1, hy1, hx2, hy2 = cord[0], cord[1], cord[0]+cord[2], cord[1]+cord[3]
                # px1, py1, px2, py2 = cord[0], cord[1], cord[0] + cord[2], cord[1] + cord[3]
                # hx1, hy1, hx2, hy2 = x, y, x + w, y + h
                if   (hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2):
                    is_in_oragne = True
                    break

            if not is_in_oragne:
                cv2.rectangle(with_contours,(x,y),(x+w,y+h),(0,255,0),2)
                apple = apple + 1
        else:
            cv2.rectangle(with_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)
            apple += 1
# with_contours = cv2.drawContours(org,rest,-1,(255,255,0),3)
print(banana,orange,apple)
while key != ord('q'):
    cv2.imshow('result', org)
    # cv2.imshow('result', with_contours)
    key = cv2.waitKey(30)
# TODO: Implement detection method.
