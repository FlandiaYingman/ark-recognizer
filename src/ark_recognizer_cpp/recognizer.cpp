//
// Created by Flandia on 2021/12/11.
//

#include "item.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

std::vector<FullItem> load_items_from_file() {
    std::ifstream fiis("items/data/item_table.json");
    auto full_items = load_full_items(fiis);
    return full_items;
}

std::vector<Item> load_items_from_file(const std::vector<FullItem> &full_items) {
    std::ifstream iis("items/data/item_icon_table.json");
    auto items = load_items(iis, full_items);
    return items;
}


int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cerr << "usage: \n./recognizer <scene_filename...> " << std::endl;
        return 0;
    }

    const auto filenames = std::vector<std::string>(argv + 1, argv + argc);

    std::vector<FullItem> full_items = load_items_from_file();
    for (auto &item: full_items) std::cout << item.to_string() << ", ";
    std::cout << "\n\n";

    std::vector<Item> items = load_items_from_file(full_items);
    for (auto &item: items) std::cout << item.to_string() << ", ";
    std::cout << "\n\n";

    return 0;
}