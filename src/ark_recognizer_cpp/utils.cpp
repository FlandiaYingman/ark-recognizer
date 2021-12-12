//
// Created by Flandia on 2021/12/12.
//

#include "utils.h"


hash_t average_hash(const cv::Mat &image, int hash_size) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(hash_size, hash_size));

    hash_t diff;
    diff.reserve(hash_size * hash_size);
    auto mean = cv::mean(image)[0];
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            auto pixel = resized.at<uint8_t>(i, j);
            diff.push_back(pixel > mean);
        }
    }

    return diff;
}

int hamming_distance(const hash_t &hash1, const hash_t &hash2) {
    int counter = 0;
    if (hash1.size() != hash2.size()) throw std::invalid_argument("hash1.size() != hash2.size()");
    size_t size = hash1.size();
    for (size_t i = 0; i < size; ++i) {
        if (hash1[i] != hash2[i]) counter++;
    }
    return counter;
}

void adjustBrightnessContrast(const cv::Mat &in, cv::Mat &out, double brightness, double contrast) {
    auto b = brightness / 100.0;
    auto c = contrast / 100.0;
    auto k = tan((45 + 44 * c) / 180 * CV_PI);

    in.copyTo(out);
    out.forEach<uchar>([&](uchar &u, const int *position) -> void {
        u = (uchar) round(std::clamp((u - 127.5 * (1 - b)) * k + 127.5 * (1 + b) + 127.5, 0.0, 255.0));
    });
}

void cvtToGray(const cv::Mat &in, cv::Mat &out) {
    switch (in.channels()) {
        case 3:
            cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
            break;
        case 4:
            cv::cvtColor(in, out, cv::COLOR_BGRA2GRAY);
            break;
        default:
            in.copyTo(out);
            break;
    }
}


