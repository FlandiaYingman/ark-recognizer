//
// Created by Flandia on 2021/12/11.
//

#include "recognize_icon.h"
#include <vector>
#include <cmath>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

double estimate_circle_radius(const Size &size) {
    double w = size.width, h = size.height;
    if (w / h > 16.0 / 9.0) {
        w = h * (16.0 / 9.0);
    }
    double estimate = w / 10.0;
    return estimate / 2;
}


vector<IRResult> recognize_icon(Mat scene_image) {
    cvtColor(scene_image, scene_image, COLOR_BGR2GRAY);

    auto estimate = estimate_circle_radius(scene_image.size());
    auto estimate_low = (int) round(estimate * 0.9);
    auto estimate_high = (int) round(estimate * 1.1);
    vector<Vec3f> circles;
    HoughCircles(scene_image, circles,
                 HOUGH_GRADIENT, 1, estimate, 50, 30,
                 estimate_low, estimate_high);
    vector<IRResult> ir_results;

    std::transform(circles.begin(), circles.end(),
                   std::back_inserter(ir_results),
                   [&](Vec3f &circle) -> IRResult { return {circle[0], circle[1], estimate}; });
    return ir_results;
}
