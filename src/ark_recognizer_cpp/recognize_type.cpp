//
// Created by Flandia on 2021/12/12.
//

#include <opencv2/imgproc.hpp>
#include "recognize_type.h"

using namespace std;
using namespace cv;

std::map<std::string, cv::Mat> preprocess_item_templates(const std::vector<Item> &items) {
    map<string, Mat> templs;
    for (const auto &item: items) {
        auto templ = item.item_icon.clone();
        cvtColor(templ, templ, COLOR_BGRA2BGR);
        resize(templ, templ, Size(), TM_SCALE, TM_SCALE, INTER_CUBIC);
        templs[item.item_id] = templ;
    }
    return templs;
}

std::map<std::string, cv::Mat> preprocess_item_template_masks(const std::vector<Item> &items) {
    map<string, Mat> masks;
    for (const auto &item: items) {
        vector<cv::Mat> channels;
        cv::split(item.item_icon.clone(), channels);

        auto mask = channels[3];
        threshold(mask, mask, 170, 255, THRESH_BINARY);
        resize(mask, mask, Size(), TM_SCALE, TM_SCALE, INTER_CUBIC);
        masks[item.item_id] = mask;
    }
    return masks;
}

TRResult match_item_template(const Mat &icon_image,
                             const Item &item,
                             const Mat &templ, const Mat &mask) {
    auto icon_w = icon_image.cols, icon_h = icon_image.rows;
    auto templ_w = templ.cols, templ_h = templ.rows;
    if (icon_w < templ_w || icon_h < templ_h) return TRResult{item, Point(), Size(), -HUGE_VAL};

    Mat result;
    matchTemplate(icon_image, templ, result, TM_CCOEFF_NORMED, mask);
    double max_val;
    Point max_loc;
    minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);

    auto tr_result = TRResult{item, max_loc, templ.size(), max_val};
    return tr_result;
}

TRResult match_all_item_templates(const Mat &icon_image,
                                  const vector<Item> &items,
                                  map<string, Mat> templs,
                                  map<string, Mat> masks) {
    vector<TRResult> results;
    for (const auto &item: items) {
        auto templ = templs[item.item_id];
        auto mask = masks[item.item_id];
        auto result = match_item_template(icon_image, item, templ, mask);
        results.push_back(result);
    }

    auto result = *max_element(
            results.begin(), results.end(),
            [&](TRResult &tr1, TRResult &tr2) -> bool { return tr1.val < tr2.val; }
    );
    return result;
}

std::vector<TRResult> recognize_type(const cv::Mat &scene_image, const std::vector<IRResult> &ir_results,
                                     const std::vector<Item> &items,
                                     const std::map<std::string, cv::Mat> &templates,
                                     const std::map<std::string, cv::Mat> &masks) {
    vector<TRResult> results;
    for (const auto &ir_result: ir_results) {
        double circle_x = ir_result.x, circle_y = ir_result.y, circle_radius = ir_result.radius;
        double icon_side_length = circle_radius * (TEMPL_ICON_SIDE_LENGTH / TEMPL_CIRCLE_RADIUS) * 1.1;

        int w = scene_image.cols, h = scene_image.rows;
        int x0 = clamp((int) round(circle_x - icon_side_length / 2), 0, w);
        int x1 = clamp((int) round(circle_x + icon_side_length / 2), 0, w);
        int y0 = clamp((int) round(circle_y - icon_side_length / 2), 0, h);
        int y1 = clamp((int) round(circle_y + icon_side_length / 2), 0, h);

        auto icon_image = scene_image(Range(y0, y1), Range(x0, x1));
        if (icon_image.empty()) continue;

        auto icon_scale_factor = (TEMPL_CIRCLE_RADIUS / circle_radius) * TM_SCALE;
        resize(icon_image, icon_image, Size(), icon_scale_factor, icon_scale_factor);

        auto result = match_all_item_templates(icon_image, items, templates, masks);
        if (result.val < TM_THRESHOLD) continue;

        result.loc = Point((int) round(x0 + result.loc.x / icon_scale_factor),
                           (int) round(y0 + result.loc.y / icon_scale_factor));
        result.size = Size((int) round(result.size.width / icon_scale_factor),
                           (int) round(result.size.height / icon_scale_factor));

        results.push_back(result);
    }
    return results;
}
