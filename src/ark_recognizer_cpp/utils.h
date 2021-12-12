//
// Created by Flandia on 2021/12/12.
//

#ifndef ARK_RECOGNIZER_CPP_UTILS_H
#define ARK_RECOGNIZER_CPP_UTILS_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

typedef std::vector<bool> hash_t;

void cvtToGray(const cv::Mat &in, cv::Mat &out);

void adjustBrightnessContrast(const cv::Mat &in, cv::Mat &out, double brightness, double contrast);

hash_t average_hash(const cv::Mat &image, int hash_size = 16);

int hamming_distance(const hash_t &hash1, const hash_t &hash2);

#endif //ARK_RECOGNIZER_CPP_UTILS_H
