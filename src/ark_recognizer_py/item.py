import json
from dataclasses import dataclass
from typing import Dict

import cv2 as cv
from numpy.typing import NDArray


@dataclass
class Item:
    """
    Represents an item in Arknights. Items have their own item IDs and icons.
    """
    item_id: str
    item_name: str
    item_icon: NDArray

    def __str__(self):
        return "%s(%s)" % (self.item_name, self.item_id)

    def __repr__(self):
        return self.__str__()


def load_item_name_id_dict():
    with open("items/data/item_table.json", "r") as item_data_file:
        item_data = json.load(item_data_file)
    dict_name_id = {}
    for item_obj in item_data['items'].values():
        item_id = item_obj["itemId"]
        item_name = item_obj["name"]
        dict_name_id[item_name] = item_id
    return dict_name_id


ITEM_DICT_NAME_ID: Dict[str, str] = load_item_name_id_dict()


def load_items():
    with open("items/data/item_icon_table.json", "r") as item_data_file:
        item_data = json.load(item_data_file)
    items = []
    for item_obj in item_data:
        item_name = item_obj["name"]
        item_id = ITEM_DICT_NAME_ID[item_name]
        image_id = item_obj["id"]
        image = cv.imread("items/icon/%s.png" % image_id, cv.IMREAD_UNCHANGED)
        items.append(Item(item_id, item_name, image))
    return items


ITEMS: list[Item] = load_items()
