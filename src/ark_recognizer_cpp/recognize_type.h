//
// Created by Flandia on 2021/12/12.
//

#ifndef ARK_RECOGNIZER_CPP_RECOGNIZE_TYPE_H
#define ARK_RECOGNIZER_CPP_RECOGNIZE_TYPE_H

#include "item.h"
#include "recognize_icon.h"
#include <map>
#include <string>
#include <opencv2/core/mat.hpp>

class TRResult {
public:
    Item item;
    cv::Point loc;
    cv::Size size;
    double val;
};

const double TM_SCALE = 1.0 / 3.0;
const double TM_THRESHOLD = 0.6;
const double TEMPL_ICON_SIDE_LENGTH = 183;
const double TEMPL_CIRCLE_RADIUS = 163.0 / 2.0;

std::map<std::string, cv::Mat> preprocess_item_templates(const std::vector<Item> &items);

std::map<std::string, cv::Mat> preprocess_item_template_masks(const std::vector<Item> &items);

std::vector<TRResult> recognize_type(const cv::Mat &scene_image,
                                     const std::vector<IRResult> &ir_results,
                                     const std::vector<Item> &items,
                                     const std::map<std::string, cv::Mat> &templates,
                                     const std::map<std::string, cv::Mat> &masks);

#endif //ARK_RECOGNIZER_CPP_RECOGNIZE_TYPE_H
