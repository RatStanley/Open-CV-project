import cv2
import json
import click

from pathlib import Path
from glob import glob
from tqdm import tqdm

from typing import Dict


import numpy as np




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
    # print( " " ,median_B," ",median_G," ",median_R)

    if median_B < 1 and median_G < 1 and median_R < 1:
        # print("banana", " ")
        return "banana"
    if median_B < 25 and median_B > 10 and median_G < 120 and median_G > 65  and median_R < 200 and median_R > 155:
        # print("orange", " ")
        return "orange"
    if median_B < 50 and median_B > 15 and median_G < 100 and median_G > 20 and median_R < 160  and median_R > 50:
        # print("apple", " ")
        return "apple"

    return



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
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    apple = 0
    banana = 0
    orange = 0
    org = cv2.imread(img_path)

    scale = 0.2
    size_of_view = (int(org.shape[1] * scale), int(org.shape[0] * scale))
    org = cv2.resize(org, dsize=size_of_view)

    contours, heirachy_bacground, mask = image_to_contur(img_org_=org, H=146, S=104, V=218)  # H=200,S=100,V=202
    img = cv2.bitwise_and(org, org, mask=mask)

    good_ctr = []
    for ctr in contours:
        if len(ctr) > 100:
            good_ctr.append(ctr)

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



    # TODO: Implement detection method.

    # apple = 0
    # banana = 0
    # orange = 0

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
