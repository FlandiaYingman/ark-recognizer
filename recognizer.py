import timeit

import cv2 as cv
import numpy as np
# The image border (in pixels percentage) of the item number in the item icon. For example, assume there's a result
# item image queried from a scene image. To extract the digit part of the image, crop ~67% from top, etc.
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy import ndarray

from recognizer_feature_matching import detect_features
from recognizer_feature_matching import match_item
from recognizer_feature_matching import preprocess_items_feature
from recognizer_item import DIGIT_BORDER_BOTTOM
from recognizer_item import DIGIT_BORDER_LEFT
from recognizer_item import DIGIT_BORDER_RIGHT
from recognizer_item import DIGIT_BORDER_TOP
from recognizer_item import ITEMS
from recognizer_item import QueryItem
from recognizer_item import load_items
from recognizer_template_matching import TMResult
from recognizer_template_matching import preprocess_items_template
from recognizer_template_matching import query_items


def crop_numbers(img_match):
    img_result_h, img_result_w = img_match.shape[:2]
    img_result_btop = round(DIGIT_BORDER_TOP * img_result_h)
    img_result_bbottom = round(DIGIT_BORDER_BOTTOM * img_result_h)
    img_result_bleft = round(DIGIT_BORDER_LEFT * img_result_w)
    img_result_bright = round(DIGIT_BORDER_RIGHT * img_result_w)
    return img_match[
           img_result_btop:img_result_h - img_result_bbottom,
           img_result_bleft: img_result_w - img_result_bright
           ]


def match_items(scene_image):
    scene_image_canvas = scene_image.copy()
    if len(ITEMS) == 0:
        raise RuntimeError("ITEMS hasn't been loaded, use load_items()")
    scene_keypoints, scene_descriptor = detect_features(scene_image)
    scales = []
    for item_id, item in ITEMS.items():
        item: QueryItem
        result = match_item(item, scene_image, scene_keypoints, scene_descriptor)
        if result != -1:
            x, y, w, h, scale = result
            cv.rectangle(scene_image_canvas, (x, y), (x + w, y + w), (0, 0, 255))
            match_box = scene_image[y:y + h, x:x + h]
            scales.append(scale)

    scales = np.array(scales)
    scales = scales[np.where(scales.mean() + scales.std() / 3 > scales)]
    scales = scales[np.where(scales.mean() - scales.std() / 3 < scales)]
    return np.median(scales)


def show_query_results(scene: ndarray, query_results: list[TMResult]):
    scene_copy = scene.copy()
    for tm_result in query_results:
        cv.rectangle(scene_copy, tm_result.loc, tm_result.loc + tm_result.size, (255, 0, 0))
    image_pil = Image.fromarray(scene_copy)
    draw_pil = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("text/train_fonts/NotoSansSC-Light.otf", 18)
    for tm_result in query_results:
        draw_pil.text(tm_result.loc, str(tm_result.item), font=font, fill=(255, 0, 0))
    scene_copy = np.array(image_pil)
    return scene_copy


def recognize(scene):
    start_time = timeit.default_timer()

    scale = match_items(scene)
    print("scale factor: %f" % scale)
    tm_results = query_items(scene, 1 / scale)
    result_scene = show_query_results(scene, tm_results)

    end_time = timeit.default_timer()
    print("used %f seconds" % (end_time - start_time))
    return result_scene


def _main():
    scene_0 = cv.imread("test/scene_images/screenshot_0_0.5x.jpg", cv.IMREAD_COLOR)
    scene_1 = cv.imread("test/scene_images/screenshot_1.jpg", cv.IMREAD_COLOR)
    scene_2 = cv.imread("test/scene_images/screenshot_1.PNG", cv.IMREAD_COLOR)
    scene_3 = cv.imread("test/scene_images/screenshot_0_1x.jpg", cv.IMREAD_COLOR)

    load_items()
    preprocess_items_feature()
    preprocess_items_template()

    result_scene_0 = recognize(scene_0)
    result_scene_1 = recognize(scene_1)
    result_scene_2 = recognize(scene_2)
    result_scene_3 = recognize(scene_3)

    cv.imshow("result_scene_0", result_scene_0)
    cv.imshow("result_scene_1", result_scene_1)
    cv.imshow("result_scene_2", result_scene_2)
    cv.imshow("result_scene_3", result_scene_3)
    cv.waitKey()


if __name__ == '__main__':
    _main()
