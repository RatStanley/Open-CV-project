import cv2
import json
import click

from glob import glob

import numpy as np
from tqdm import tqdm

from typing import Dict


def empty_callback(value):
    pass


def detect_fruits(img_path: str) -> Dict[str, int]:

    cv2.namedWindow('result')
    cv2.createTrackbar('H', 'result', 0, 255, empty_callback)
    cv2.createTrackbar('S', 'result', 0, 255, empty_callback)
    cv2.createTrackbar('V', 'result', 0, 255, empty_callback)
    cv2.createTrackbar('R', 'result', 0, 255, empty_callback)
    cv2.createTrackbar('G', 'result', 0, 255, empty_callback)
    cv2.createTrackbar('B', 'result', 0, 255, empty_callback)

    img_org = cv2.imread(img_path, cv2.IMREAD_COLOR)
    scale = 0.1
    size_of_view = (int(img_org.shape[1] * scale), int(img_org.shape[0] * scale))
    img_org = cv2.resize(img_org, dsize=size_of_view)
    img_show = img_org
    img_rgb = img_org
    img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    img_hsv_converted = img_hsv
    img = np.concatenate((img_show, img_hsv, img_rgb), axis=1)
    key = ord('a')
    print(img.shape)
    while key != ord('q'):

        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        H = cv2.getTrackbarPos('H', 'result')
        S = cv2.getTrackbarPos('S', 'result')
        V = cv2.getTrackbarPos('V', 'result')
        R = cv2.getTrackbarPos('R', 'result')
        G = cv2.getTrackbarPos('G', 'result')
        B = cv2.getTrackbarPos('B', 'result')
        # H_value = img_hsv[:,:,0]
        # S_value = img_hsv[:,:,1]
        # V_value = img_hsv[:,:,2]
        # print(H_value.shape)

        ret, H_value = cv2.threshold(img_hsv[:,:,0], H, 255, cv2.THRESH_TOZERO)
        ret, S_value = cv2.threshold(img_hsv[:,:,1], S, 255, cv2.THRESH_TOZERO)
        ret, V_value = cv2.threshold(img_hsv[:,:,2], V, 255, cv2.THRESH_TOZERO)
        ret, R_value = cv2.threshold(img_org[:,:,2], R, 255, cv2.THRESH_TOZERO)
        ret, G_value = cv2.threshold(img_org[:,:,1], G, 255, cv2.THRESH_TOZERO)
        ret, B_value = cv2.threshold(img_org[:,:,0], B, 255, cv2.THRESH_TOZERO)
        img_hsv_converted[:,:,0] = H_value
        img_hsv_converted[:,:,1] = S_value
        img_hsv_converted[:,:,2] = V_value

        img_rgb[:,:,2] = R_value
        img_rgb[:,:,1] = G_value
        img_rgb[:,:,0] = B_value

        img = np.concatenate((img_show, img_hsv_converted, img_rgb), axis=1)
        cv2.imshow('result', img)
        key = cv2.waitKey(30)
    # TODO: Implement detection method.

    apple = 0
    banana = 0
    orange = 0
    print("test")

    return {'apple': apple, 'banana': banana, 'orange': orange}


detect_fruits("data/00.jpg")
