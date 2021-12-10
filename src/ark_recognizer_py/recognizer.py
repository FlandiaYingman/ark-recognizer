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
from numpy.typing import NDArray

from ark_recognizer_py import recognize_icon
from ark_recognizer_py.item import ITEMS
from ark_recognizer_py.item import Item
from ark_recognizer_py.recognize_icon import IRResult
from ark_recognizer_py.recognize_icon import recognize_icon
from ark_recognizer_py.recognize_number import NRResult
from ark_recognizer_py.recognize_number import recognize_number
from ark_recognizer_py.recognize_type import TRResult
from ark_recognizer_py.recognize_type import recognize_type


def draw_ir_results(scene_image: NDArray, ir_results: List[IRResult]):
    scene_canvas = scene_image.copy()
    if ir_results is None:
        return
    for pt in ir_results:
        a, b, r = np.rint(pt).astype(int)
        # Draw the circumference of the circle.
        cv.circle(scene_canvas, (a, b), r, (0, 0, 255))
        # Draw a small circle (of radius 1) to show the center.
        cv.circle(scene_canvas, (a, b), 1, (0, 255, 255))
    return scene_canvas


def draw_tr_results(scene_image: NDArray, tr_results: List[TRResult]):
    scene_canvas = scene_image.copy()

    if len(tr_results) <= 0:
        return scene_canvas

    for tm_result in tr_results:
        cv.rectangle(scene_canvas, tm_result.loc, tm_result.loc + tm_result.size, (255, 0, 0), lineType=cv.LINE_4)

    scene_pil = Image.fromarray(scene_canvas)
    draw_pil = ImageDraw.Draw(scene_pil)
    font = ImageFont.truetype("fonts/NotoSansSC-Regular.otf", round(tr_results[0].size[0] / 10))
    for tm_result in tr_results:
        # noinspection PyTypeChecker
        draw_pil.text(tm_result.loc, str(tm_result.item), font=font, fill=(255, 0, 0))

    # noinspection PyTypeChecker
    return np.array(scene_pil)


def draw_nr_results(scene_image: NDArray, nr_results: List[NRResult]):
    scene_canvas = scene_image.copy()
    for do_result in nr_results:
        cv.rectangle(scene_canvas, do_result.loc, do_result.loc + do_result.size, (0, 0, 255))
        cv.putText(scene_canvas, str(do_result.number), do_result.loc, cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    return scene_canvas


@dataclass
class RecognizeResult:
    """
    Represents a recognition result of a specified item.
    """
    item: Item
    number: int


def recognize(scene_image: NDArray, show=False, show_delay=500):
    scene_image_channels = scene_image.shape[2]
    if scene_image_channels == 4:
        scene_image = cv.cvtColor(scene_image, cv.COLOR_BGRA2BGR)

    start_time = timeit.default_timer()

    ir_results = recognize_icon(scene_image)
    tr_results = recognize_type(scene_image, ir_results)
    nr_results = recognize_number(scene_image, tr_results)

    end_time = timeit.default_timer()
    print("used %f seconds" % (end_time - start_time))

    if show:
        scene_canvas = scene_image.copy()
        scene_canvas = draw_ir_results(scene_canvas, ir_results)
        scene_canvas = draw_tr_results(scene_canvas, tr_results)
        scene_canvas = draw_nr_results(scene_canvas, nr_results)
        cv.imshow("results", scene_canvas)
        cv.waitKey(show_delay)

    return [RecognizeResult(do_result.item, do_result.number) for do_result in nr_results]


def merge_recognize_results(recognize_results: List[List[RecognizeResult]]) -> List[RecognizeResult]:
    recognize_results = list(itertools.chain.from_iterable(recognize_results))
    answer = {}
    for recognize_result in recognize_results:
        item = recognize_result.item
        if item.item_id in answer:
            # exists
            this_number = recognize_result.number
            that_number = answer[item.item_id].number
            if this_number != that_number:
                # number not same
                print("merge conflict: %s number, %d != %d" % (str(item), this_number, that_number),
                      file=sys.stderr)
        else:
            # not exists
            answer[item.item_id] = recognize_result
    for item in ITEMS:
        if item.item_id not in answer:
            answer[item.item_id] = RecognizeResult(item, 0)
    return list(answer.values())


def export_penguin_planner_config(recognize_results: List[RecognizeResult]) -> str:
    json_obj = {'@type': "@penguin-statistics/planner/config",
                'items': [{'id': it.item.item_id, 'have': it.number}
                          for it in recognize_results]}
    return json.dumps(json_obj)


def _main():
    files = [sys.argv[i] for i in range(1, len(sys.argv))]
    recognize_results = [recognize(cv.imread(file)) for file in files]
    merged_recognize_results = merge_recognize_results(recognize_results)

    penguin_planner_config = export_penguin_planner_config(merged_recognize_results)
    print(penguin_planner_config)


if __name__ == '__main__':
    _main()
