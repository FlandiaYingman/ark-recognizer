import json

import cv2 as cv
import numpy as np

DIGIT_BORDER_TOP = 116 / 173  # =67.05%
DIGIT_BORDER_BOTTOM = 21 / 173  # =~12.14%
DIGIT_BORDER_LEFT = 56 / 172  # =~32.56%
DIGIT_BORDER_RIGHT = 28 / 172  # =~16.28%


class QueryItem:

    def __extract_alpha_mask(self):
        alpha = self.image[:, :, 3]
        return alpha

    def __extract_item_number_mask(self):
        width, height = (self.image.shape[1], self.image.shape[0])
        btop = round(DIGIT_BORDER_TOP * height)
        bbottom = round(DIGIT_BORDER_BOTTOM * height)
        bleft = round(DIGIT_BORDER_LEFT * width)
        bright = round(DIGIT_BORDER_RIGHT * width)
        mask = np.zeros((183, 183), np.uint8)
        pt1 = (bleft, btop)
        pt2 = (183 - bright, 183 - bbottom)
        mask = cv.rectangle(mask, pt1, pt2, 255, -1)
        mask = cv.bitwise_not(mask)
        return mask

    def __init__(self, item_id, item_name, image):
        self.item_id = item_id
        self.item_name = item_name
        self.image = image
        _, self.image_mask = cv.threshold(
            cv.bitwise_and(
                self.__extract_alpha_mask(),
                self.__extract_item_number_mask()),
            127, 255, cv.THRESH_BINARY
        )
        self.image = self.image[:, :, :3]

        self.width, self.height = (self.image.shape[1], self.image.shape[0])

        self.keypoints = None
        self.descriptor = None

        self.template = None
        self.template_mask = None

    def __str__(self):
        return "%s(%s)" % (self.item_name, self.item_id)

    def __repr__(self):
        return self.__str__()


ITEMS: dict[str, QueryItem] = {}


def load_items():
    global ITEMS
    with open("items/data/items.json", "r") as item_data_file:
        item_data = json.load(item_data_file)
    for i in item_data:
        image = cv.imread("items/icon/%s.png" % i["id"], cv.IMREAD_UNCHANGED)
        ITEMS[i["id"]] = QueryItem(i["id"], i["name"], image)


class QueryResult:

    def __init__(self, item, locs, vals, size):
        self.item = item
        self.locs = locs
        self.vals = vals
        self.size = size

    def __str__(self):
        return "Query Result %s: %s, %s, %f" % (self.item, self.locs, self.vals, self.size)

    def pt1(self):
        return self.loc

    def pt2(self):
        return (self.loc[0] + self.size[0], self.loc[1] + self.size[1])


QueryResults = dict[str, QueryResult]
