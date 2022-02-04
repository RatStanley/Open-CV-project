import cv2
import copy
import numpy as np
import statistics
from matplotlib import pyplot as plt
import os

# img_org = cv2.imread("data/03.jpg", cv2.IMREAD_COLOR)
# scale = 0.2
# size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
# img_org = cv2.resize(img_org, dsize=size_of_view)


#######################################################################################################################


def show_img(show):
    # show = zamaskowane
    scale = 0.2
    size_of_view = (int(show.shape[1] * scale), int(show.shape[0] * scale))
    cv2.namedWindow('result')
    key = ord('a')
    show = cv2.resize(show, dsize=size_of_view)
    while key != ord('q'):
        cv2.imshow('result', show)
        # plt.show()
        # cv2.imshow('result', with_contours)
        key = cv2.waitKey(30)


# maski obiektów
data = ["data/00.jpg", "data/01.JPG", "data/02.JPG", "data/03.JPG", "data/04.JPG", "data/05.JPG", "data/06.JPG",
        "data/07.JPG", "data/08.JPG", "data/09.JPG"]
answer = [[2,2,2],[0,0,3],[1,1,3],[1,0,2],[1,1,2],[0,2,3],[1,1,2],[1,1,2],[2,2,2],[1,2,3]]

bad = 0
for path in range(len(data)):
    apple = 0
    banana = 0
    orange = 0
    img = cv2.imread(data[path], cv2.IMREAD_COLOR)
    HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    print(np.median(HSV[:, :, 0]), " ", np.median(HSV[:, :, 1]), " " , np.median(HSV[:, :, 2]))
    maska = img.copy()
    HSV = cv2.medianBlur(HSV, 31)
    ret, maska[:, :, 0] = cv2.threshold(HSV[:, :, 0], 150+int(np.median(HSV[:, :, 0])), 255, cv2.THRESH_BINARY)
    ret, maska[:, :, 1] = cv2.threshold(HSV[:, :, 1], 140+int(np.median(HSV[:, :, 1])), 255, cv2.THRESH_BINARY)
    ret, maska[:, :, 2] = cv2.threshold(HSV[:, :, 2], 20+int(np.median(HSV[:, :, 2])), 255, cv2.THRESH_BINARY)

    # ret, maska[:, :, 0] = cv2.threshold(HSV[:, :, 0], 248, 255, cv2.THRESH_BINARY)
    # ret, maska[:, :, 1] = cv2.threshold(HSV[:, :, 1], 120, 255, cv2.THRESH_BINARY)
    # ret, maska[:, :, 2] = cv2.threshold(HSV[:, :, 2], 211, 255, cv2.THRESH_BINARY)

    maska = cv2.cvtColor(maska, cv2.COLOR_RGB2GRAY)
    maska[maska != 0] = 255
    zamaskowane = cv2.bitwise_and(img, img, mask=maska)
    kontury, hierarchia = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for kontur in kontury:
        x, y, w, h = cv2.boundingRect(kontur)
        if w > 300 and h > 300:
            x = x + int(1/4*w)
            w = int(1/2*w)
            y = y + int(1 / 4 * h)
            h = int(1 / 2 * h)
            mediana_kolorów = np.min(zamaskowane[y:y + h, x:x + w])
            prostokat = HSV[y:y + h, x:x + w]
            H = np.median(prostokat[:,:,0])
            S = np.median(prostokat[:,:,1])
            V = np.median(prostokat[:, :, 2])
            if 0 < H and H < 180 and 140 < S and S < 210 and 50 < V and V < 200:
                apple = apple +1
            elif 10 < H and H < 20 and 210 < S and S < 250 and 150 < V and V < 250:
                orange = orange +1

            elif 18 < H and H < 30 and 10 < S and S < 70 and 130 < V and V < 190:
                banana = banana + 1


    print("ban ", banana, "or ", orange, "ap ", apple)
    if not (answer[path] == [banana,orange,apple]):
        print("err")
        bad += 1
    show_img(zamaskowane)
print("błąd -> ",bad/10)
    #
#

# TODO: Implement detection method.
