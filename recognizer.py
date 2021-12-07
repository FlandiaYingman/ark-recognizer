import os
import timeit

import cv2 as cv
import numpy as np
# The image border (in pixels percentage) of the item number in the item icon. For example, assume there's a result
# item image queried from a scene image. To extract the digit part of the image, crop ~67% from top, etc.
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy import ndarray

import template_matching
from circle_detection import detect_circle
from template_matching import TMResult


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
        cv.rectangle(scene_canvas, tm_result.loc, tm_result.loc + tm_result.size, (255, 0, 0))

    scene_pil = Image.fromarray(scene_canvas)
    draw_pil = ImageDraw.Draw(scene_pil)
    font = ImageFont.truetype("text/train_fonts/NotoSansSC-Light.otf", round(tm_results[0].size[0] / 12))
    for tm_result in tm_results:
        draw_pil.text(tm_result.loc, str(tm_result.item), font=font, fill=(255, 0, 0))

    return np.array(scene_pil)


def recognize(scene_image):
    start_time = timeit.default_timer()

    cd_results = detect_circle(scene_image)
    cv.imshow("cd_result", draw_circles(scene_image, cd_results))

    tm_results = template_matching.match_all(scene_image, cd_results)
    cv.imshow("tm_result", draw_tm_results(scene_image, tm_results))

    end_time = timeit.default_timer()
    print("used %f seconds" % (end_time - start_time))

    cv.waitKey()


def _main():
    # for file in os.listdir("test/screenshots/"):
    #     recognize(cv.imread("test/screenshots/%s" % file))
    recognize(cv.imread("test/scene_images/498704999.jpg"))


if __name__ == '__main__':
    _main()
