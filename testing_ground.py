import cv2
import json
import click

from glob import glob
from tqdm import tqdm

from typing import Dict

def empty_callback(value):
    pass

def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    cv2.createTrackbar('R', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('G', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('B', 'image', 0, 255, empty_callback)


    img_org = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_path, cv2.IMREAD_COLOR)
    key = ord('a')
    while key != ord('q'):


        cv2.imshow('result', img_edges)
        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(30)
    # TODO: Implement detection method.

    apple = 0
    banana = 0
    orange = 0
    print("test")

    return {'apple': apple, 'banana': banana, 'orange': orange}




detect_fruits("data/00.jpg")
