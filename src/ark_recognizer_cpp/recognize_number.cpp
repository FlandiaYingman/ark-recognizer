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
                       const std::map<std::string, std::string> &hash_map) {
    ostringstream ss;
    for (const auto &digit_image: digit_images) {
        auto digit_image_hash = average_hash(digit_image);

        vector<int> distances;
        distances.reserve(10);
        for (int i = 0; i < 10; ++i) {
            distances.push_back(hamming_distance(
                    hex2hash(hash_map.at(to_string(i))),
                    digit_image_hash)
            );
        }

        auto min_distance_iter = min_element(distances.begin(), distances.end());
        auto min_distance_index = min_distance_iter - distances.begin();

        if (*min_distance_iter <= 64) {
            ss << min_distance_index;
        }
    }

    return stoi(ss.str());
}


std::vector<NRResult> recognize_number(const cv::Mat &scene_image,
                                       const std::vector<TRResult> &tr_results,
                                       const std::map<std::string, std::string> &hash_map) {
    vector<NRResult> nr_results;
    for (const auto &tr_result: tr_results) {
        auto icon_image = scene_image(Rect(tr_result.loc, tr_result.size));
        resize(icon_image, icon_image, ICON_BASE_SIZE);

        auto number_region = icon_image(Rect(NUMBER_REGION_LOC, NUMBER_REGION_SIZE));

        auto digit_images = find_digit_images(number_region);
        auto number = parse_digit_images(digit_images, hash_map);

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

std::array<std::string, 10> load_digit_hashes(istream &in) {
    nlohmann::json json_list;
    in >> json_list;

    std::array<string, 10> arr;
    size_t i = 0;
    for (const auto &hash_json_obj: json_list) {
        auto hash_str = hash_json_obj.get<string>();
        arr[i++] = hash_str;
    }

    return arr;
}

hash_t hex2hash(const string &hex) {
    hash_t hash_val;
    for (const auto &ch: hex) {
        int val = stoi(string{ch}, nullptr, 16);
        hash_val.push_back(val & (1 << 3));
        hash_val.push_back(val & (1 << 2));
        hash_val.push_back(val & (1 << 1));
        hash_val.push_back(val & (1 << 0));
    }
    return hash_val;
}

std::string hash2hex(const hash_t &hash) {
    ostringstream oss;
    oss << std::hex;
    for (size_t i = 0; i < hash.size(); i += 4) {
        int num = hash[i + 0] * (1 << 3) +
                  hash[i + 1] * (1 << 2) +
                  hash[i + 2] * (1 << 1) +
                  hash[i + 3] * (1 << 0);
        oss << num;
    }
    return oss.str();
}
