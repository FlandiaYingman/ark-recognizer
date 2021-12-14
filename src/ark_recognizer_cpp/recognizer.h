//
// Created by Flandia on 2021/12/14.
//

#ifndef ARK_RECOGNIZER_CPP_RECOGNIZER_H
#define ARK_RECOGNIZER_CPP_RECOGNIZER_H

#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "item.h"
#include "recognize_icon.h"
#include "recognize_type.h"
#include "recognize_number.h"

class RecognizeResult {
public:
    Item item;
    int number = -1;
};

[[maybe_unused]]
static void to_json(nlohmann::json &j, const RecognizeResult &o) {
    j = nlohmann::json{
            {"id",   o.item.item_id},
            {"have", o.number},
    };
}


void load_item(const std::string &item_id, const cv::Mat &icon);

void load_hash(const std::string &name, const std::string &hash);


std::vector<RecognizeResult> recognize(const std::vector<cv::Mat> &scene_images);

std::string export_penguin_arkplanner_json(const std::vector<RecognizeResult> &results);


#endif //ARK_RECOGNIZER_CPP_RECOGNIZER_H
