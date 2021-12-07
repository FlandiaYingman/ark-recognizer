import json
from dataclasses import dataclass

import cv2 as cv
from numpy.typing import NDArray


@dataclass
class Item:
    """
    Represents an item in Arknights. Items have their own item IDs and icons.
    """
    item_id: int
    item_name: str
    item_icon: NDArray

    def __str__(self):
        return "%s(%s)" % (self.item_name, self.item_id)

    def __repr__(self):
        return self.__str__()


def load_items():
    with open("items/data/items.json", "r") as item_data_file:
        item_data = json.load(item_data_file)
    items = []
    for item_obj in item_data:
        item_id = int(item_obj["id"])
        item_name = item_obj["name"]
        items.append(load_item(item_id, item_name))
    return items


def load_item(item_id, item_name):
    image = cv.imread("items/icon/%s.png" % item_id, cv.IMREAD_UNCHANGED)
    return Item(item_id, item_name, image)


ITEMS: list[Item] = load_items()


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
