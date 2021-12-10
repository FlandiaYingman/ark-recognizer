from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy.typing import NDArray

from ark_recognizer_py.item import Item
from ark_recognizer_py.recognize_type import TRResult
from ark_recognizer_py.utils import adjust_brightness_contrast
from ark_recognizer_py.utils import convert_to_gray
from ark_recognizer_py.utils import crop
from ark_recognizer_py.utils import hamming_distance
from ark_recognizer_py.utils import hash_image


@dataclass
class NRResult:
    item: Item
    number: int
    loc: NDArray
    size: NDArray


# When the icon size is 216x216 pixels:
#  - the number height is 26 pixels;
#  - the number region is located at (88, 144)
#  - the number region size is (92, 44)
#  - the border size (distance between number and number region) is 9 pixels;

ICON_BASE_SIZE = np.array([216, 216])
TEXT_HEIGHT = 26
TEXT_WIDTH = 19
NUMBER_REGION_LOC = np.array([88, 144])
NUMBER_REGION_SIZE = np.array([92, 44])
BORDER_SIZE = 9


def find_digit_images(number_region: NDArray):
    digit_images = []

    number_region = convert_to_gray(number_region)
    number_region = adjust_brightness_contrast(number_region, brightness=-20, contrast=60)

    _, thresh_number_region = cv.threshold(number_region, 64, 256, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresh_number_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_boxes = [cv.boundingRect(contour) for contour in contours]
    contour_boxes.sort(key=lambda it: it[0])

    for contour_box in contour_boxes:
        [x, y, w, h] = contour_box
        if (TEXT_WIDTH + 3) >= w >= (TEXT_WIDTH - 9) and (TEXT_HEIGHT + 6) >= h >= (TEXT_HEIGHT - 6):
            digit_image = crop(number_region, np.array([x, y]), np.array([w, h]))
            digit_images.append(digit_image)

    return digit_images


def parse_digit_images(digit_images: List[NDArray]):
    digits_str = ""
    for digit_image in digit_images:
        digit_image_hash = hash_image(digit_image)

        distances = [hamming_distance(digit_image_hash, x) for x in DIGITS_HASH]
        min_distance = min(distances)
        min_distance_index = distances.index(min_distance)

        if min_distance <= 64:
            digits_str += str(min_distance_index)

    return int(digits_str or "-1")


def recognize_number(scene_image: NDArray, tr_results: List[TRResult]):
    nr_results = []
    for tr_result in tr_results:
        icon_image = crop(scene_image, tr_result.loc, tr_result.size)
        icon_image = cv.resize(icon_image, ICON_BASE_SIZE)
        number_region = crop(icon_image, NUMBER_REGION_LOC, NUMBER_REGION_SIZE)

        digits = find_digit_images(number_region)
        number = parse_digit_images(digits)

        loc = np.rint(tr_result.loc + (tr_result.size * NUMBER_REGION_LOC / ICON_BASE_SIZE)).astype(int)
        size = np.rint(tr_result.size * NUMBER_REGION_SIZE / ICON_BASE_SIZE).astype(int)

        nr_result = NRResult(tr_result.item, number, loc, size)
        nr_results.append(nr_result)
    return nr_results


def __generate_digits():
    DIGITS = "0123456789"

    estimate_font_size = 1
    while True:
        font = ImageFont.truetype("fonts/NotoSansSC-Regular.otf", estimate_font_size)
        l, t, r, b = font.getbbox(DIGITS)
        if b - t >= TEXT_HEIGHT:
            break
        else:
            estimate_font_size += 1

    canvas = Image.new("L", (r + 2 * BORDER_SIZE, (b - t) + 2 * BORDER_SIZE))
    white = 255

    draw = ImageDraw.Draw(canvas)
    draw.text((BORDER_SIZE, BORDER_SIZE - t), DIGITS, font=font, fill=white)

    # noinspection PyTypeChecker
    np_canvas = np.array(canvas)
    return np_canvas


def __preprocess_digits_hash():
    digits_image = __generate_digits()
    digits_hash = [hash_image(digit_image) for digit_image in find_digit_images(digits_image)]
    assert len(digits_hash) == 10
    return digits_hash


DIGITS_HASH: List[NDArray] = __preprocess_digits_hash()
