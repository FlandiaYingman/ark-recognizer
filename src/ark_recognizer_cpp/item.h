//
// Created by Flandia on 2021/12/11.
//

#ifndef ARK_RECOGNIZER_CPP_ITEM_H
#define ARK_RECOGNIZER_CPP_ITEM_H

#include <string>
#include <sstream>
#include <opencv2/core/mat.hpp>

/**
 * Represents a item from Arknights with its icon.
 */
class Item {
public:
    std::string item_id;
    std::string item_name;
    cv::Mat item_icon;

    [[maybe_unused]] [[nodiscard]]
    std::string to_string() const {
        std::ostringstream out;
        out << item_name << "(" << item_id << ")";
        return out.str();
    }

};

#endif //ARK_RECOGNIZER_CPP_ITEM_H
