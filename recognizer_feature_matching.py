import json
from math import sqrt

import cv2 as cv
import numpy as np

SIFT = cv.SIFT_create()


def detect_features(img, mask=None):
    global SIFT
    keypoints, descriptor = SIFT.detectAndCompute(img, mask, None)
    return keypoints, descriptor


def show_features(img, keypoints):
    img_keypoints = cv.drawKeypoints(img, keypoints, None)
    cv.imshow(str(img), img_keypoints)


def extract_alpha_mask(img):
    retval, mask = cv.threshold(img[:, :, 3], 0, 255, cv.THRESH_BINARY)
    return mask


class QueryItem:

    def __init__(self, item_id, item_name, image):
        self.item_id = item_id
        self.item_name = item_name
        self.image = image
        self.mask = extract_alpha_mask(image)
        self.keypoints, self.descriptor = detect_features(self.image, self.mask)

    def __str__(self):
        return "%s(%s)" % (self.item_name, self.item_id)

    def show(self):
        show_features(self.image, self.keypoints)


ITEMS = {}


def load_items():
    global ITEMS
    with open("items/data/items.json", "r") as item_data_file:
        item_data = json.load(item_data_file)
    for i in item_data:
        image = cv.imread("items/icon/%s.png" % i["id"], cv.IMREAD_UNCHANGED)
        ITEMS[i["id"]] = QueryItem(i["id"], i["name"], image)


BF_MATCHER = cv.BFMatcher_create()
MIN_MATCH_COUNT = 10


def match_item(item: QueryItem, scene_image, scene_keypoints, scene_descriptor):
    query_image = item.image
    query_keypoints = item.keypoints
    query_descriptor = item.descriptor

    matches = BF_MATCHER.knnMatch(query_descriptor, scene_descriptor, k=2)
    good_matches = [m for m, n in matches if (m.distance / n.distance) < 0.75]
    if len(good_matches) < MIN_MATCH_COUNT:
        print("%s: no enough good matches, required %d but was %d" % (item, MIN_MATCH_COUNT, len(good_matches)))
        return -1

    query_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    scene_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    transformation, mask = cv.findHomography(query_pts, scene_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()  # ?

    if transformation is None:
        print("%s: no transformation is found with %d good matches" % (item, len(good_matches)))
        return -1

    query_image_height, query_image_width, _ = query_image.shape
    query_rect = np.float32(
        [[0, 0],
         [0, query_image_height - 1],
         [query_image_width - 1, query_image_height - 1],
         [query_image_width - 1, 0]]
    ).reshape(-1, 1, 2)
    scene_rect = cv.perspectiveTransform(query_rect, transformation)

    match_rect = np.int32(scene_rect.reshape(4, 2))
    x, y, w, h = cv.boundingRect(match_rect)
    scene_h, scene_w, _ = scene_image.shape

    if x < 0 or y < 0 or x+w > scene_w or y+h > scene_h:
        print("%s: transformation invalid (out of bounds) with %d good matches" % (item, len(good_matches)))
        return -1

    scale = sqrt(w * h) / sqrt(query_image_width * query_image_height)

    print("%s: match with %d good matches, (%d, %d, %d, %d) scale %f" % (item, len(good_matches), x, y, w, h, scale))
    return x, y, w, h, scale