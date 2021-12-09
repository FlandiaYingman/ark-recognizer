from dataclasses import dataclass

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from circle_detection import CDResult
from circle_detection import ITEM_ICON_OUT_CIRCLE_RADIUS
from item import ITEMS
from item import Item
from utils import clamp
from utils import img_size


@dataclass
class TMResult:
    """
    Represents a result of a template matching.
    """
    item: Item
    size: NDArray
    loc: NDArray
    val: float

    def pt1(self):
        return self.loc

    def pt2(self):
        return self.loc + self.size

    def crop(self, scene_image: NDArray):
        x_begin, y_begin = self.loc
        x_end, y_end = self.loc + self.size
        return scene_image[y_begin:y_end, x_begin:x_end, :]


TM_TEMPLATE_SCALE = 1 / 3
TM_THRESHOLD = 0.6


def _preprocess_items_template():
    items_template = {}
    for item in ITEMS:
        template = item.item_icon
        template = cv.cvtColor(template, cv.COLOR_BGRA2BGR)
        template = cv.resize(template, None,
                             fx=TM_TEMPLATE_SCALE,
                             fy=TM_TEMPLATE_SCALE,
                             interpolation=cv.INTER_CUBIC)
        items_template[item.item_id] = template
    return items_template


def _preprocess_items_templates_mask():
    items_template_mask = {}
    for item in ITEMS:
        mask = item.item_icon[:, :, 3]
        _, mask = cv.threshold(mask, 170, 255, cv.THRESH_BINARY)
        mask = cv.resize(mask, None,
                         fx=TM_TEMPLATE_SCALE,
                         fy=TM_TEMPLATE_SCALE,
                         interpolation=cv.INTER_CUBIC)
        items_template_mask[item.item_id] = mask
    return items_template_mask


ITEMS_TEMPLATE: dict[str, NDArray] = _preprocess_items_template()
ITEMS_TEMPLATE_MASK: dict[str, NDArray] = _preprocess_items_templates_mask()


def match_item(scene: NDArray, circle: CDResult, item: Item) -> TMResult | None:
    cx, cy, cr = np.rint(circle).astype(int)
    scale = ITEM_ICON_OUT_CIRCLE_RADIUS / cr
    cr = cr / 0.75

    w, h = img_size(scene)
    x0 = clamp(round(cx - cr), 0, w)
    x1 = clamp(round(cx + cr), 0, w)
    y0 = clamp(round(cy - cr), 0, h)
    y1 = clamp(round(cy + cr), 0, h)

    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None

    scene_crop = scene[y0:y1, x0:x1]
    scene_crop = cv.resize(scene_crop, None, fx=scale * TM_TEMPLATE_SCALE, fy=scale * TM_TEMPLATE_SCALE)

    tmpl = ITEMS_TEMPLATE[item.item_id]
    tmpl_mask = ITEMS_TEMPLATE_MASK[item.item_id]

    scene_crop_w, scene_crop_h = img_size(scene_crop)
    tmpl_w, tmpl_h = img_size(tmpl)
    if scene_crop_h < tmpl_h or scene_crop_w < tmpl_w:
        return None

    result: NDArray = cv.matchTemplate(scene_crop,
                                       templ=tmpl,
                                       mask=tmpl_mask,
                                       method=cv.TM_CCOEFF_NORMED
                                       )
    result = np.where(np.isfinite(result), result, -1)
    size = np.rint(np.array((tmpl.shape[1], tmpl.shape[0])) / (scale * TM_TEMPLATE_SCALE)).astype(int)

    _, max_val, _, max_loc = cv.minMaxLoc(result)
    val = max_val
    loc = np.rint((np.array(max_loc) / (scale * TM_TEMPLATE_SCALE)) + (x0, y0)).astype(int)
    tm_result = TMResult(item, size, loc, val)
    return tm_result


def match_items(scene: NDArray, circle: CDResult) -> TMResult | None:
    results = []
    for item in ITEMS:
        result = match_item(scene, circle, item)
        if result is not None:
            results.append(result)
    if len(results) <= 0:
        return None
    results.sort(key=lambda x: x.val, reverse=True)
    result = results[0]
    if result.val >= TM_THRESHOLD:
        return result
    else:
        return None


def match_all(scene: NDArray, circles: list[CDResult]) -> list[TMResult]:
    results = []
    for circle in circles:
        result = match_items(scene, circle)
        if result is not None:
            results.append(result)
    return results
