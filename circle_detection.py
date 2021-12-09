from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from numpy.typing import NDArray

from item import ITEMS
from utils import img_size

ITEM_ICON_SIZE = 183
ITEM_ICON_OUT_CIRCLE_RADIUS = 163 / 2
ITEM_ICON_MID_CIRCLE_RADIUS = 151 / 2
ITEM_ICON_IN_CIRCLE_RADIUS = 139 / 2


@dataclass
class CDResult:
    """
    Represents a result of a Circle Detection.
    """
    x: float
    y: float
    radius: float


def _preprocess_items_radius():
    items_cd = {}
    for item in ITEMS:
        # use image_mask, because only the very outer circle is wanted
        item_icon = item.item_icon[:, :, 3]
        item_icon_height = item_icon.shape[0]
        item_icon = cv2.Canny(item_icon, 300, 500)
        circles = cv2.HoughCircles(item_icon,
                                   cv2.HOUGH_GRADIENT, 1, round(item_icon_height / 2), param1=50, param2=10,
                                   minRadius=round(item_icon_height / 2.5),
                                   maxRadius=round(item_icon_height / 1.5))

        if circles is not None:
            radius: float = np.concatenate(circles[:, :, 2]).mean()
            items_cd[item.item_id] = radius
    return items_cd


ITEMS_RADIUS: Dict[str, float] = _preprocess_items_radius()
ITEMS_CD_RADIUS: float = np.array(list(ITEMS_RADIUS.values())).mean()


def estimate_circle_radius(size: NDArray):
    w, h = size
    # if the aspect ratio > 16:9, increasing the width will not increase the icon radius
    if w / h > 16 / 9:
        w = h * (16 / 9)
    # the icon radius is always 1/10 of the screen width
    estimate = w / 10
    # radius wanted, not diameter
    return estimate / 2


def detect_circle(scene_image):
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

    estimate = estimate_circle_radius(img_size(scene_image))
    estimate_low = round(estimate * 0.9)
    estimate_high = round(estimate * 1.1)
    circles = cv2.HoughCircles(scene_image,
                               cv2.HOUGH_GRADIENT, 1, round(estimate), param1=50, param2=30,
                               minRadius=estimate_low,
                               maxRadius=estimate_high)
    circles = circles.reshape(-1, 3)
    for circle in circles:
        # set radius to estimated radius
        circle[2] = estimate
    return circles
