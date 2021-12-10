import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from ark_recognizer_py.item import ITEMS
from ark_recognizer_py.item import Item
from ark_recognizer_py.recognize_icon import IRResult
from ark_recognizer_py.utils import clamp
from ark_recognizer_py.utils import img_size


@dataclass
class TRResult:
    """
    Represents a type recognition result.
    """
    item: Item
    size: NDArray
    loc: NDArray
    val: float


TM_SCALE = 1 / 3
TM_THRESHOLD = 0.6
TEMPL_ICON_SIDE_LENGTH = 183
TEMPL_CIRCLE_RADIUS = 163 / 2


def __preprocess_items_template():
    items_template = {}
    for item in ITEMS:
        template = item.item_icon
        template = cv.cvtColor(template, cv.COLOR_BGRA2BGR)
        template = cv.resize(template, None,
                             fx=TM_SCALE,
                             fy=TM_SCALE,
                             interpolation=cv.INTER_CUBIC)
        items_template[item.item_id] = template
    return items_template


def __preprocess_items_templates_mask():
    items_template_mask = {}
    for item in ITEMS:
        mask = item.item_icon[:, :, 3]
        _, mask = cv.threshold(mask, 170, 255, cv.THRESH_BINARY)
        mask = cv.resize(mask, None,
                         fx=TM_SCALE,
                         fy=TM_SCALE,
                         interpolation=cv.INTER_CUBIC)
        items_template_mask[item.item_id] = mask
    return items_template_mask


ITEMS_TEMPLATE: dict[str, NDArray] = __preprocess_items_template()
ITEMS_TEMPLATE_MASK: dict[str, NDArray] = __preprocess_items_templates_mask()


def match_item_template(icon_image: NDArray, item: Item) -> TRResult:
    tmpl = ITEMS_TEMPLATE[item.item_id]
    tmpl_mask = ITEMS_TEMPLATE_MASK[item.item_id]

    icon_w, icon_h = img_size(icon_image)
    tmpl_w, tmpl_h = img_size(tmpl)
    if icon_h < tmpl_h or icon_w < tmpl_w:
        return TRResult(item, np.zeros(0), np.zeros(0), -math.inf)

    result: NDArray = cv.matchTemplate(icon_image,
                                       templ=tmpl,
                                       mask=tmpl_mask,
                                       method=cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(result)
    tr_result = TRResult(item, img_size(tmpl), np.array(max_loc), max_val)
    return tr_result


def match_all_item_templates(icon_image: NDArray) -> TRResult:
    # match icon image with all item templates
    results = []
    for item in ITEMS:
        result = match_item_template(icon_image, item)
        results.append(result)

    result = max(results, key=lambda x: x.val)
    return result


def recognize_type(scene_image: NDArray, circles: list[IRResult]) -> list[TRResult]:
    results = []
    for circle in circles:
        # the x, y, and radius of this icon circle
        circle_x, circle_y, circle_radius = np.rint(circle).astype(int)
        # the side length of this icon. for tolerant, * 1.1
        icon_side_length = circle_radius * (TEMPL_ICON_SIDE_LENGTH / TEMPL_CIRCLE_RADIUS) * 1.1

        w, h = img_size(scene_image)
        x0 = clamp(round(circle_x - icon_side_length / 2), 0, w)
        x1 = clamp(round(circle_x + icon_side_length / 2), 0, w)
        y0 = clamp(round(circle_y - icon_side_length / 2), 0, h)
        y1 = clamp(round(circle_y + icon_side_length / 2), 0, h)
        # the icon image cropped from the scene image
        icon_image = scene_image[y0:y1, x0:x1]

        # check if the icon cropped is empty
        if len(icon_image) == 0:
            continue

        # the scale factor to scale this icon size to template icon size
        icon_scale_factor = (TEMPL_CIRCLE_RADIUS / circle_radius) * TM_SCALE
        # the icon image cropped and resized from scene image
        icon_image = cv.resize(icon_image, None, fx=icon_scale_factor, fy=icon_scale_factor)

        result = match_all_item_templates(icon_image)
        if result.val < TM_THRESHOLD:
            continue

        # calculate the actual size and location in the scene image
        result.size = np.rint(result.size / icon_scale_factor).astype(int)
        result.loc = np.rint((x0, y0) + result.loc / icon_scale_factor).astype(int)

        results.append(result)

    return results
