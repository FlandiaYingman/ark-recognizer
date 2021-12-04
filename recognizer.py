import timeit

import cv2 as cv
import numpy as np

# The image border (in pixels percentage) of the item number in the item icon. For example, assume there's a result
# item image queried from a scene image. To extract the digit part of the image, crop ~67% from top, etc.
from examples.recognizer_digit_ocr_example import generate_samples_responses
from recognizer_digit_ocr import find_digits
from recognizer_digit_ocr import generate_train_image
from recognizer_digit_ocr import match_digit
from recognizer_digit_ocr import obtain_model
from recognizer_feature_matching import ITEMS
from recognizer_feature_matching import QueryItem
from recognizer_feature_matching import detect_features
from recognizer_feature_matching import load_items
from recognizer_feature_matching import match_item
from recognizer_feature_matching import show_features

DIGIT_BORDER_TOP = 116 / 173  # =67.05%
DIGIT_BORDER_BOTTOM = 21 / 173  # =~12.14%
DIGIT_BORDER_LEFT = 56 / 172  # =~32.56%
DIGIT_BORDER_RIGHT = 28 / 172  # =~16.28%


def sift_orb_bounding_box():
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_query, des_query = sift.detectAndCompute(img_query, None)
    kp_scene, des_scene = sift.detectAndCompute(scene, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    # matches = bf.match(des_query, des_scene)
    matches = bf.knnMatch(des_query, des_scene, k=2)
    # # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Find bounding box
    src_pts = np.float32([kp_query[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h, w = img_query.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv.perspectiveTransform(pts, M)

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    )

    # cv.drawMatchesKnn expects list of lists as matches.
    result = cv.drawMatchesKnn(img_query, kp_query, scene, kp_scene, good, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    result = cv.polylines(result, [np.int32(dst + (w, 0))], True, (0, 0, 255), None, cv.LINE_AA)

    rect = np.int32(dst.reshape(4, 2))
    x, y, w, h = cv.boundingRect(rect)

    img_result = scene[y:y + h, x:x + w]

    return img_result, result


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
    show_features(scene_image, scene_keypoints)
    for item_id, item in ITEMS.items():
        item: QueryItem
        result = match_item(item, scene_image, scene_keypoints, scene_descriptor)
        if result != -1:
            x, y, w, h, scale = result
            cv.rectangle(scene_image_canvas, (x, y), (x+w, y+w), (0, 0, 255))
            match_box = scene_image[y:y + h, x:x + h]
            # cv.imshow(None, match_box)
            # cv.waitKey()
    cv.imshow(None, scene_image_canvas)
    cv.waitKey()


if __name__ == '__main__':
    scene = cv.imread("test/scene_images/screenshot_0_1x.jpg", cv.IMREAD_COLOR)
    load_items()
    show_features(ITEMS["100035"].image, ITEMS["100035"].keypoints)
    start_time = timeit.default_timer()
    match_items(scene)
    end_time = timeit.default_timer()
    print("used %f seconds" % (end_time - start_time))
