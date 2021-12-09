import math
from dataclasses import dataclass
from typing import List

import cv2 as cv
import imagehash
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from imagehash import ImageHash
from numpy.typing import NDArray

from item import Item
from template_matching import TMResult

"""
When the template size is 216x216 pixels:
 - the text height is 26 pixels;
 - the border size is 9 pixels;
 - the text region is located at (88, 144)
 - the text region size is (92, 44)
"""

TEMPLATE_SIZE = np.array([216, 216])
TEXT_REGION_POS = np.array([88, 144])
TEXT_REGION_SIZE = np.array([92, 44])
TEXT_HEIGHT = 26
TEXT_WIDTH = 19
BORDER_SIZE = 9


def adjust_brightness_contrast(image: NDArray, brightness, contrast):
    b = brightness / 100.0
    c = contrast / 100.0

    k = math.tan((45 + 44 * c) / 180 * math.pi)

    image = image.astype(np.float64)
    image = (image - 127.5 * (1 - b)) * k + 127.5 * (1 + b)
    image = np.clip(image, 0, 255)
    image = np.rint(image)
    image = image.astype(np.uint8)
    return image


def find_digits(img):
    digits = []

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = adjust_brightness_contrast(img, brightness=-20, contrast=60)
    ret, thresh_img = cv.threshold(img, 64, 256, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_boxes = list(map(lambda c: cv.boundingRect(c), contours))
    contours_boxes.sort(key=lambda b: b[0])
    for contour_box in contours_boxes:
        [x, y, w, h] = contour_box
        if (TEXT_WIDTH + 3) >= w >= (TEXT_WIDTH - 9) and (TEXT_HEIGHT + 6) >= h >= (TEXT_HEIGHT - 6):
            crop_img = img[y:y + h, x:x + w]
            digits.append(crop_img)

            # rect_img = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)
            # rect_img = cv.rectangle(rect_img, (x, y), (x + w, y + h), color=(0, 0, 255))
            # rect_img = cv.resize(rect_img, None, fx=10.0, fy=10.0, interpolation=cv.INTER_NEAREST)
            # cv.imshow(None, rect_img)
            # cv.waitKey()
    return digits


def hash_image(image: NDArray):
    image_pil = Image.fromarray(image)
    image_hash = imagehash.average_hash(image_pil, 16)
    return image_hash


def round_odd(n):
    answer = round(n)
    if answer % 2 == 1:
        return answer
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return answer + 1
    else:
        return answer - 1


def crop_text_region(img_match):
    x, y = TEXT_REGION_POS
    w, h = TEXT_REGION_SIZE
    return img_match[y: y + h, x: x + w]


NUMBERS = "0123456789"


def draw_digits():
    estimate_font_size = 1
    while True:
        font = ImageFont.truetype("train_fonts/NotoSansSC-Regular.otf", estimate_font_size)
        l, t, r, b = font.getbbox(NUMBERS)
        if b - t >= TEXT_HEIGHT:
            break
        else:
            estimate_font_size += 1

    canvas = Image.new("L", (r + 2 * BORDER_SIZE, (b - t) + 2 * BORDER_SIZE))
    white = 255

    draw = ImageDraw.Draw(canvas)
    draw.text((BORDER_SIZE, BORDER_SIZE - t), NUMBERS, font=font, fill=white)

    # noinspection PyTypeChecker
    np_canvas = np.array(canvas)
    return np_canvas


def preprocess_digits_hash():
    digits_hash = []
    digits_image = draw_digits()
    for digit_image in find_digits(digits_image):
        digits_hash.append(hash_image(digit_image))
    if len(digits_hash) != 10:
        raise RuntimeError
    return digits_hash


DIGITS_HASH: List = preprocess_digits_hash()


def hamming_distance(hash1: ImageHash, hash2: ImageHash):
    return sum(c1 != c2 for c1, c2 in zip(hash1.hash.reshape(-1), hash2.hash.reshape(-1)))


def parse_digits(digits_image: List[NDArray]):
    digits_str = ""
    for digit_image in digits_image:
        digit_image_hash = hash_image(digit_image)

        distances = [hamming_distance(digit_image_hash, x) for x in DIGITS_HASH]
        min_distance = min(distances)
        min_distance_index = distances.index(min_distance)

        if min_distance <= 64:
            digits_str += str(min_distance_index)
    if len(digits_str) <= 0:
        return -1
    return int(digits_str)


@dataclass
class DOResult:
    item: Item
    quantity: int
    loc: NDArray
    size: NDArray


def parse_quantities(scene_image: NDArray, tm_results: List[TMResult]):
    do_results = []
    for tm_result in tm_results:
        scene_template = tm_result.crop(scene_image)
        scene_template = cv.resize(scene_template, TEMPLATE_SIZE)
        text_region = crop_text_region(scene_template)

        digits = find_digits(text_region)
        quantity = parse_digits(digits)

        loc = np.rint(tm_result.loc + (tm_result.size * TEXT_REGION_POS / 216)).astype(int)
        size = np.rint(tm_result.size * TEXT_REGION_SIZE / 216).astype(int)

        do_results.append(DOResult(tm_result.item, quantity, loc, size))
    return do_results
