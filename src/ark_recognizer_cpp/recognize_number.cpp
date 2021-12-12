//
// Created by Flandia on 2021/12/12.
//

#include <opencv2/imgproc.hpp>
#include "recognize_number.h"
#include "recognize_type.h"
#include <nlohmann/json.hpp>
#include "utils.h"

using namespace std;
using namespace cv;

vector<Mat> find_digit_images(Mat &number_region) {
    vector<Mat> digit_images;

    cvtToGray(number_region, number_region);
    adjustBrightnessContrast(number_region, number_region, -20, 60);

    Mat thresh_number_region;
    threshold(number_region, thresh_number_region, 64, 256, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresh_number_region, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Rect> contour_boxes;
    transform(contours.begin(), contours.end(), back_inserter(contour_boxes),
              [](vector<Point> &v) -> Rect { return boundingRect(v); });
    sort(contour_boxes.begin(), contour_boxes.end(),
         [](Rect &r1, Rect &r2) -> bool { return r1.x < r2.x; });

    for (const auto &contour_box: contour_boxes) {
        int w = contour_box.width, h = contour_box.height;
        if (TEXT_WIDTH + 3 >= w && w >= TEXT_WIDTH - 9 && TEXT_HEIGHT + 6 >= h && h >= TEXT_HEIGHT - 6) {
            auto digit_image = number_region(contour_box);
            digit_images.push_back(digit_image);
        }
    }
    return digit_images;
}

int parse_digit_images(const vector<Mat> &digit_images,
                       const array<hash_t, 10> &digit_hashes) {
    ostringstream ss;
    for (const auto &digit_image: digit_images) {
        auto digit_image_hash = average_hash(digit_image);

        vector<int> distances;
        distances.reserve(digit_hashes.size());
        for (const auto &item: digit_hashes) {
            distances.push_back(hamming_distance(item, digit_image_hash));
        }

        auto min_distance_iter = min_element(distances.begin(), distances.end());
        auto min_distance_index = min_distance_iter - distances.begin();

        if (*min_distance_iter <= 64) {
            ss << min_distance_index;
        }
    }

    return stoi(ss.str());
}


vector<NRResult> recognize_number(const Mat &scene_image,
                                  const vector<TRResult> &tr_results,
                                  const array<hash_t, 10> &digit_hashes) {
    vector<NRResult> nr_results;
    for (const auto &tr_result: tr_results) {
        auto icon_image = scene_image(Rect(tr_result.loc, tr_result.size));
        resize(icon_image, icon_image, ICON_BASE_SIZE);

        auto number_region = icon_image(Rect(NUMBER_REGION_LOC, NUMBER_REGION_SIZE));

        auto digit_images = find_digit_images(number_region);
        auto number = parse_digit_images(digit_images, digit_hashes);

        auto loc = tr_result.loc + (Point(
                (int) round((double) tr_result.size.width * NUMBER_REGION_LOC.x / ICON_BASE_SIZE.width),
                (int) round((double) tr_result.size.height * NUMBER_REGION_LOC.y / ICON_BASE_SIZE.height)
        ));
        auto size = Size(
                (int) round((double) tr_result.size.width * NUMBER_REGION_SIZE.width / ICON_BASE_SIZE.width),
                (int) round((double) tr_result.size.height * NUMBER_REGION_SIZE.height / ICON_BASE_SIZE.height)
        );

        auto nr_result = NRResult{tr_result.item, number, loc, size};
        nr_results.push_back(nr_result);
    }
    return nr_results;
}

std::array<hash_t, 10> load_digit_hashes(istream &in) {
    nlohmann::json json_list;
    in >> json_list;

    std::array<hash_t, 10> arr;
    size_t i = 0;
    for (const auto &hash_json_obj: json_list) {
        auto hash_str = hash_json_obj.get<string>();
        hash_t hash_val;
        for (const auto &ch: hash_str) {
            int val = stoi(string{ch}, nullptr, 16);
            hash_val.push_back(val & (1 << 3));
            hash_val.push_back(val & (1 << 2));
            hash_val.push_back(val & (1 << 1));
            hash_val.push_back(val & (1 << 0));
        }
        arr[i++] = hash_val;
    }

    return arr;
}
