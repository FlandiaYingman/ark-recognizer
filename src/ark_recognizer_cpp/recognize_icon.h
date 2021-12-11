//
// Created by Flandia on 2021/12/11.
//

#ifndef ARK_RECOGNIZER_CPP_RECOGNIZE_ICON_H
#define ARK_RECOGNIZER_CPP_RECOGNIZE_ICON_H

#include <vector>
#include <opencv2/core.hpp>

/**
 * Represents an icon recognition result.
 */
class IRResult {
public:
    double x;
    double y;
    double radius;
};

std::vector<IRResult> recognize_icon(cv::Mat scene_image);


#endif //ARK_RECOGNIZER_CPP_RECOGNIZE_ICON_H
