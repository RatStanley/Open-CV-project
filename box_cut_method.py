import cv2
import copy
import numpy as np
import statistics
from  matplotlib import pyplot as plt
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
    ret, H_value = cv2.threshold(H_Orgin, H, 255, cv2.THRESH_BINARY)
    ret, S_value = cv2.threshold(S_Orgin, S, 255, cv2.THRESH_BINARY)
    ret, V_value = cv2.threshold(V_Orgin, V, 255, cv2.THRESH_BINARY)
    img[:, :, 0] = H_value
    img[:, :, 1] = S_value
    img[:, :, 2] = V_value


    img = cv2.medianBlur(img, 31)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[img != 0] = 255

    contur_list, heirachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contur_list, heirachy, img



def color_rec(img):
    median_B = np.median(img[:, :, 0])
    median_G = np.median(img[:, :, 1])
    median_R = np.median(img[:, :, 2])
    print( " " ,median_B," ",median_G," ",median_R)

    if median_B < 1 and median_G < 1 and median_R < 1:
        print("banana", " ")
        return "banana"
    if median_B < 25 and median_B > 10 and median_G < 120 and median_G > 65  and median_R < 200 and median_R > 155:
        print("orange", " ")
        return "orange"
    if median_B < 50 and median_B > 15 and median_G < 100 and median_G > 20 and median_R < 160  and median_R > 50:
        print("apple", " ")
        return "apple"

    return


#maski obiektów
apple = 0
banana = 0
orange = 0
org = cv2.imread("data/09.JPG")

scale = 0.2
size_of_view = (int(org.shape[1] * scale), int(org.shape[0] * scale))
org = cv2.resize(org, dsize=size_of_view)

contours, heirachy_bacground, mask = image_to_contur(img_org_=org, H=146, S=104, V=218)#H=200,S=100,V=202
img = cv2.bitwise_and(org,org,mask = mask)

good_ctr = []
for ctr in contours:
    if len(ctr) > 100:
        good_ctr.append(ctr)




# x,y,w,h = cv2.boundingRect(contours[3])
#
#
#
# test = img[y:y+h,x:x+w]
#
# sum_B = np.sum(test[:,:,0])
# sum_G = np.sum(test[:,:,1])
# sum_R = np.sum(test[:,:,2])
# print(sum_B)
# print(sum_G)
# print(sum_R)

# print(test.mean())
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([test],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([3,256])
#     plt.ylim([0, 1500])
# # plt.show()
it = 0

# print("")
for crt in good_ctr:
    x, y, w, h = cv2.boundingRect(crt)
    temp_image = img[y:y + h, x:x + w]
    temp_obj = color_rec(temp_image)
    if temp_obj == "banana":
        banana += 1
    if temp_obj == "orange":
        orange += 1
    if temp_obj == "apple":
        apple += 1
# for crt in good_ctr:
#     # x,y,w,h = cv2.boundingRect(crt)
#     x, y, w, h = cv2.boundingRect(crt)
#     temp_image = img[y:y + h, x:x + w]
#     # color_rec(temp_image)
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
#     img = cv2.putText(img, str(it), (x,y), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (255,0,0), 2, cv2.LINE_AA)
#     it += 1

print("ban ",banana,"or ", orange,"ap ",apple)
while key != ord('q'):
    cv2.imshow('result', img)
    # plt.show()
    # cv2.imshow('result', with_contours)
    key = cv2.waitKey(30)
# TODO: Implement detection method.
