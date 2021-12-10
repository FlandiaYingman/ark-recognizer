import json
import unittest
from pathlib import Path
from typing import List

import cv2

from ark_recognizer_py.recognizer import RecognizeResult
from ark_recognizer_py.recognizer import merge_recognize_results
from ark_recognizer_py.recognizer import recognize

SCENE_IMAGES = "test_images/scene_images"
SCENE_DATA = "test_data/scene_data"

SHOW = False
SHOW_DELAY = None


def results_to_dict(r: List[RecognizeResult]):
    return {i.item.item_name: i.number for i in r}


def scene_images_test(base: str, show=SHOW, show_delay=SHOW_DELAY):
    print("test: %s" % base)
    files = [str(file) for file in Path(SCENE_IMAGES).glob("%s*" % base)]
    results = [recognize(cv2.imread(file), show=show, show_delay=show_delay) for file in files]
    results = merge_recognize_results(results)

    actual_json = json.dumps(results_to_dict(results),
                             sort_keys=True,
                             ensure_ascii=False)
    print("actual json: \n%s" % actual_json)
    expected_json = json.dumps(json.loads((Path(SCENE_DATA) / ("%s.json" % base)).read_text()),
                               sort_keys=True,
                               ensure_ascii=False)
    print("expected json: \n%s" % expected_json)

    expected = json.loads(expected_json)
    actual = json.loads(actual_json)
    return expected, actual


class Case1920x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("1920x1080")
        self.assertEqual(expected, actual)


class Case1350x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("1350x1080")
        self.assertEqual(expected, actual)


class Case1440x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("1440x1080")
        self.assertEqual(expected, actual)


class Case1620x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("1620x1080")
        self.assertEqual(expected, actual)


class Case1728x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("1728x1080")
        self.assertEqual(expected, actual)


class Case2160x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("2160x1080")
        self.assertEqual(expected, actual)


class Case2520x1080(unittest.TestCase):
    def test(self):
        expected, actual = scene_images_test("2520x1080")
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
