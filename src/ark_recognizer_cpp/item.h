//
// Created by Flandia on 2021/12/11.
//

#ifndef ARK_RECOGNIZER_CPP_ITEM_H
#define ARK_RECOGNIZER_CPP_ITEM_H

#include <string>
#include <sstream>
#include <opencv2/core/mat.hpp>

/**
 * Represents an item from Arknights.
 */
class FullItem {
public:
    std::string item_id;
    std::string item_name;
    int sort_id;
    // ...

    [[nodiscard]]
    std::string to_string() const;
};

std::vector<FullItem> load_full_items(std::istream &in);


/**
 * Represents a simplified item from Arknights with its icon.
 */
class Item {
public:
    std::string item_id;
    std::string item_name;
    cv::Mat item_icon;

    [[nodiscard]]
    std::string to_string() const;
};

std::vector<Item> load_items(std::istream &in, const std::vector<FullItem> &full_items);


#endif //ARK_RECOGNIZER_CPP_ITEM_H
