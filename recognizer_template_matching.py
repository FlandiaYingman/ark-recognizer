import math
from dataclasses import dataclass
from numbers import Number

import cv2 as cv
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from recognizer_item import ITEMS
from recognizer_item import QueryItem


def convert_template(image: ndarray) -> ndarray:
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.resize(image, None,
                      fx=TM_TEMPLATE_SCALING_FACTOR,
                      fy=TM_TEMPLATE_SCALING_FACTOR,
                      interpolation=cv.INTER_CUBIC)
    return image


def convert_template_mask(image_mask: ndarray) -> ndarray:
    image_mask = cv.resize(image_mask, None,
                           fx=TM_TEMPLATE_SCALING_FACTOR,
                           fy=TM_TEMPLATE_SCALING_FACTOR,
                           interpolation=cv.INTER_CUBIC)
    return image_mask


def preprocess_items_template():
    for item in ITEMS.values():
        if item.template is None:
            item.template = convert_template(item.image)
        if item.template_mask is None:
            item.template_mask = convert_template_mask(item.image_mask)


@dataclass
class TMResult:
    item: QueryItem
    size: NDArray
    val: Number
    loc: NDArray

    def pt1(self):
        return self.loc

    def pt2(self):
        return self.loc + self.size


def query_item(item: QueryItem, screenshot_scaled: NDArray) -> list[TMResult]:
    # Choose TM_CCOEFF instead of TM_SQDIFF & TM_CCORR, because the others don't punish wrong values.
    result: NDArray = cv.matchTemplate(screenshot_scaled,
                                       templ=item.template,
                                       mask=item.template_mask,
                                       method=cv.TM_CCORR_NORMED
                                       )
    result = np.where(np.isfinite(result), result, -1)
    tm_result_list = []
    size = np.array((item.template.shape[1], item.template.shape[0]))
    while True:
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        val = max_val
        loc = np.array(max_loc)
        if val > TM_THRESHOLD:
            tm_result_list.append(TMResult(item, size, val, loc))
            cv.rectangle(result, loc - size, loc + size, color=0.0, thickness=-1)
        else:
            break
    return tm_result_list


TM_THRESHOLD = 0.875
TM_TEMPLATE_SCALING_FACTOR = 1 / 3


def query_items(screenshot: ndarray, factor: float) -> list[TMResult]:
    factor *= TM_TEMPLATE_SCALING_FACTOR
    screenshot = screenshot.copy()
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
    screenshot = cv.resize(screenshot, None, fx=factor, fy=factor, interpolation=cv.INTER_CUBIC)

    tm_results_all = []
    for item in ITEMS.values():
        tm_results = query_item(item, screenshot)
        tm_results_all.extend(tm_results)
        print(tm_results)

    for result in tm_results_all:
        result.loc = np.rint(result.loc / factor).astype(int)
        result.size = np.rint(result.size / factor).astype(int)

    tm_results_all.sort(key=lambda x: x.val, reverse=True)
    for i in range(0, len(tm_results_all)):
        result_i = tm_results_all[i]
        for j in range(i + 1, len(tm_results_all)):
            result_j = tm_results_all[j]
            if math.isnan(result_j.val):
                continue
            if overlap(result_i, result_j):
                result_j.val = math.nan
    tm_results_all = list(filter(lambda x: not math.isnan(x.val), tm_results_all))

    return tm_results_all


def overlap(r1: TMResult, r2: TMResult):
    if r1.pt1()[0] > r2.pt2()[0] or r1.pt2()[0] < r2.pt1()[0]:
        return False
    if r1.pt1()[1] > r2.pt2()[1] or r1.pt2()[1] < r2.pt1()[1]:
        return False
    return True
