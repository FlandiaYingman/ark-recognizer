//
// Created by Flandia on 2021/12/12.
//

#ifndef ARK_RECOGNIZER_CPP_RECOGNIZE_NUMBER_H
#define ARK_RECOGNIZER_CPP_RECOGNIZE_NUMBER_H

#include "item.h"
#include "utils.h"
#include "recognize_type.h"

class NRResult {
public:
    Item item;
    int number;
    cv::Point loc;
    cv::Size size;
};

// When the icon size is 216x216 pixels:
//  - the number height is 26 pixels;
//  - the number region is located at (88, 144)
//  - the number region size is (92, 44)
//  - the border size (distance between number and number region) is 9 pixels;

const cv::Size ICON_BASE_SIZE(216, 216); // NOLINT(cert-err58-cpp)
const int TEXT_HEIGHT = 26;
const int TEXT_WIDTH = 19;
const cv::Point NUMBER_REGION_LOC(88, 144); // NOLINT(cert-err58-cpp)
const cv::Size NUMBER_REGION_SIZE(92, 44); // NOLINT(cert-err58-cpp)
//const int BORDER_SIZE = 9;


std::array<hash_t, 10> load_digit_hashes(std::istream &in);


std::vector<NRResult> recognize_number(const cv::Mat &scene_image,
                                       const std::vector<TRResult> &tr_results,
                                       const std::array<hash_t, 10> &digit_hashes);

#endif //ARK_RECOGNIZER_CPP_RECOGNIZE_NUMBER_H
