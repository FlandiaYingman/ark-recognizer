import math

import cv2
import imagehash
import numpy as np
from PIL import Image
from numpy.typing import NDArray


def img_size(arr: NDArray):
    return np.array([arr.shape[1], arr.shape[0]])


def extract_alpha(item):
    return item.item_icon[:, :, 3]


def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n


def crop(image: NDArray, loc: NDArray, size: NDArray):
    x_begin, y_begin = loc
    x_end, y_end = loc + size
    return image[y_begin:y_end, x_begin:x_end]


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


def hamming_distance(hash1: NDArray, hash2: NDArray):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def hash_image(image: NDArray):
    image_pil = Image.fromarray(image)
    image_hash = imagehash.average_hash(image_pil, 16)
    return image_hash.hash.reshape(-1)


def convert_to_gray(image: NDArray) -> NDArray:
    if len(image.shape) == 2:
        return image
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image
