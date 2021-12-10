import itertools
import json
import sys
import timeit
from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy import ndarray
from numpy.typing import NDArray

from ark_recognizer_py import circle_detection
from ark_recognizer_py.digit_ocr import DOResult
from ark_recognizer_py.digit_ocr import parse_quantities
from ark_recognizer_py.item import ITEMS
from ark_recognizer_py.item import Item
from ark_recognizer_py.template_matching import TMResult
from ark_recognizer_py.template_matching import match_all


def draw_circles(scene_image, circles):
    scene_canvas = scene_image.copy()
    if circles is None:
        return
    for pt in circles:
        a, b, r = np.rint(pt).astype(int)
        # Draw the circumference of the circle.
        cv.circle(scene_canvas, (a, b), r, (0, 0, 255))
        # Draw a small circle (of radius 1) to show the center.
        cv.circle(scene_canvas, (a, b), 1, (0, 255, 255))
    return scene_canvas


def draw_tm_results(scene: ndarray, tm_results: list[TMResult]):
    scene_canvas = scene.copy()

    if len(tm_results) <= 0:
        return scene_canvas

    for tm_result in tm_results:
        cv.rectangle(scene_canvas, tm_result.loc, tm_result.loc + tm_result.size, (255, 0, 0), lineType=cv.LINE_4)

    scene_pil = Image.fromarray(scene_canvas)
    draw_pil = ImageDraw.Draw(scene_pil)
    font = ImageFont.truetype("fonts/NotoSansSC-Regular.otf", round(tm_results[0].size[0] / 10))
    for tm_result in tm_results:
        # noinspection PyTypeChecker
        draw_pil.text(tm_result.loc, str(tm_result.item), font=font, fill=(255, 0, 0))

    # noinspection PyTypeChecker
    return np.array(scene_pil)


def draw_do_results(scene_image: NDArray, do_results: List[DOResult]):
    scene_canvas = scene_image.copy()
    for do_result in do_results:
        cv.rectangle(scene_canvas, do_result.loc, do_result.loc + do_result.size, (0, 0, 255))
        cv.putText(scene_canvas, str(do_result.quantity), do_result.loc, cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    return scene_canvas


@dataclass
class RecognizeResult:
    """
    Represents a recognition result of a specified item.
    """
    item: Item
    quantity: int

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def recognize(scene_image: NDArray, show=False):
    scene_image_channels = scene_image.shape[2]
    if scene_image_channels == 4:
        scene_image = cv.cvtColor(scene_image, cv.COLOR_BGRA2BGR)

    start_time = timeit.default_timer()

    cd_results = circle_detection.detect_circle(scene_image)
    tm_results = match_all(scene_image, cd_results)
    do_results = parse_quantities(scene_image, tm_results)

    end_time = timeit.default_timer()
    print("used %f seconds" % (end_time - start_time))

    if show:
        scene_canvas = scene_image.copy()
        scene_canvas = draw_circles(scene_canvas, cd_results)
        scene_canvas = draw_tm_results(scene_canvas, tm_results)
        scene_canvas = draw_do_results(scene_canvas, do_results)
        cv.imshow("results", scene_canvas)
        cv.waitKey()

    return [RecognizeResult(do_result.item, do_result.quantity) for do_result in do_results]


def merge_recognize_results(recognize_results: List[RecognizeResult] | List[List[RecognizeResult]]) \
        -> List[RecognizeResult]:
    if all([isinstance(result, List) for result in recognize_results]):
        recognize_results = list(itertools.chain.from_iterable(recognize_results))
    answer = {}
    for recognize_result in recognize_results:
        item = recognize_result.item
        if item.item_id in answer:
            # exists
            this_quantity = recognize_result.quantity
            that_quantity = answer[item.item_id].quantity
            if this_quantity != that_quantity:
                # quantity not same
                print("merge conflict: %s quantity, %d != %d" % (str(item), this_quantity, that_quantity),
                      file=sys.stderr)
        else:
            # not exists
            answer[item.item_id] = recognize_result
    for item in ITEMS:
        if item.item_id not in answer:
            answer[item.item_id] = RecognizeResult(item, 0)
    return list(answer.values())


def _main():
    files = [sys.argv[i] for i in range(1, len(sys.argv))]
    recognize_results = [recognize(cv.imread(file)) for file in files]
    merged_recognize_results = merge_recognize_results(flatten_recognize_results)

    penguin_planner_config = export_penguin_planner_config(merged_recognize_results)
    print(penguin_planner_config)


def export_penguin_planner_config(recognize_results: List[RecognizeResult]) -> str:
    json_obj = {'@type': "@penguin-statistics/planner/config",
                'items': [{'id': it.item.item_id, 'have': it.quantity}
                          for it in recognize_results]}
    return json.dumps(json_obj)


if __name__ == '__main__':
    _main()
