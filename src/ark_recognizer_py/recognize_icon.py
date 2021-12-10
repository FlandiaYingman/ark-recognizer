from dataclasses import dataclass

import cv2
from numpy.typing import NDArray

from ark_recognizer_py.utils import img_size


@dataclass
class IRResult:
    """
    Represents an icon recognition result.
    """
    x: float
    y: float
    radius: float


def estimate_circle_radius(scene_size: NDArray):
    w, h = scene_size
    # if the aspect ratio > 16:9, increasing the width will not increase the icon radius
    if w / h > 16 / 9:
        w = h * (16 / 9)
    # the icon radius is always 1/10 of the width
    estimate = w / 10
    # radius wanted, not diameter
    return estimate / 2


def recognize_icon(scene_image: NDArray):
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

    estimate = estimate_circle_radius(img_size(scene_image))
    estimate_low = round(estimate * 0.9)
    estimate_high = round(estimate * 1.1)
    circles = cv2.HoughCircles(scene_image,
                               cv2.HOUGH_GRADIENT, 1, round(estimate), param1=50, param2=30,
                               minRadius=estimate_low,
                               maxRadius=estimate_high)
    # cv2.HoughCircles() returns a 3D array.
    # however, the 1st dimension contains only 1 element and is completely redundant.
    circles = circles.reshape(-1, 3)
    # replace the radius from cv2.HoughCircles to estimated
    for circle in circles:
        circle[2] = estimate
    return circles
