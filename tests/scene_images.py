import json
import os
import unittest
from typing import Dict
from typing import List

import cv2

from ark_recognizer_py.item import ITEMS
from ark_recognizer_py.recognizer import RecognizeResult
from ark_recognizer_py.recognizer import merge_recognize_results
from ark_recognizer_py.recognizer import recognize

DIR = "test_images/scene_images"


def results_to_dict(r: List[RecognizeResult]):
    return {i.item.item_name: i.number for i in r}


def dict_to_results(d: Dict[str, int]):
    return [RecognizeResult(next(x for x in ITEMS if x.item_name == item_name), number) for item_name, number in d]


class Case1920x1080(unittest.TestCase):
    EXPECTED_RESULT = """{"D32钢": 0, "RMA70-12": 0, "RMA70-24": 0, "三水锰矿": 0, "中级作战记录": 863, "五水研磨石": 0, "代糖": 72, 
    "先锋双芯片": 0, "先锋芯片": 13, "先锋芯片组": 0, "全新装置": 2, "凝胶": 99, "切削原液": 0, "初级作战记录": 1660, "化合切削液": 27, "医疗双芯片": 5, 
    "医疗芯片": 8, "医疗芯片组": 2, "半自然溶剂": 0, "双极纳米片": 0, "双酮": 52, "固源岩": 9, "固源岩组": 0, "基础作战记录": 1320, "基础加固建材": 0, 
    "异铁": 1, "异铁块": 0, "异铁碎片": 8, "异铁组": 0, "扭转醇": 84, "技巧概要·卷1": 1406, "技巧概要·卷2": 341, "技巧概要·卷3": 11, "提纯源岩": 0, 
    "改量装置": 0, "晶体元件": 87, "晶体电子单元": 7, "晶体电路": 1, "术师双芯片": 0, "术师芯片": 14, "术师芯片组": 0, "模组数据块": 29, "源岩": 2, 
    "炽合金": 20, "炽合金块": 1, "特种双芯片": 0, "特种芯片": 7, "特种芯片组": 2, "狙击双芯片": 0, "狙击芯片": 2, "狙击芯片组": 8, "白马醇": 0, "研磨石": 2, 
    "破损装置": 3, "碳": 883, "碳素": 121, "碳素组": 24, "精炼溶剂": 0, "糖": 131, "糖组": 12, "糖聚块": 8, "聚合凝胶": 0, "聚合剂": 0, 
    "聚酸酯": 185, "聚酸酯块": 1, "聚酸酯组": 26, "芯片助剂": 0, "装置": 2, "轻锰矿": 70, "辅助双芯片": 0, "辅助芯片": 6, "辅助芯片组": 0, "近卫双芯片": 0, 
    "近卫芯片": 5, "近卫芯片组": 10, "进阶加固建材": 0, "酮凝集": 18, "酮凝集组": 0, "酮阵列": 0, "酯原料": 97, "重装双芯片": 4, "重装芯片": 0, 
    "重装芯片组": 0, "高级作战记录": 6, "高级加固建材": 0, "龙骨": 0} """

    def test(self):
        files = [os.path.join(DIR, f) for f in os.listdir(DIR) if f.startswith("1920x1080")]
        results = [recognize(cv2.imread(file), show=False) for file in files]
        results = merge_recognize_results(results)

        results_json = json.dumps(results_to_dict(results), sort_keys=True, ensure_ascii=False)
        print(results_json)

        actual = json.loads(results_json)
        expected = json.loads(Case1920x1080.EXPECTED_RESULT)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
