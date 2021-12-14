//
// Created by Flandia on 2021/12/14.
//

#include "item.h"
#include "recognizer.h"

#include <vector>
#include <iostream>

using namespace std;

void load_items_from_file(const string &item_table_filename, const string &item_icon_table_filename) {
    ifstream it_in(item_table_filename);
    nlohmann::json item_table_json;
    it_in >> item_table_json;

    ifstream iit_in(item_icon_table_filename);
    nlohmann::json item_icon_table_json;
    iit_in >> item_icon_table_json;

    for (auto &item_json_obj: item_table_json["items"]) {
        auto item_id = item_json_obj["itemId"].get<string>();
        auto item_name = item_json_obj["name"].get<string>();

        if (item_icon_table_json.find(item_name) != item_icon_table_json.end()) {
            auto item_icon_name = item_icon_table_json[item_name].get<string>();
            auto item_icon = cv::imread("items/icon/" + item_icon_name + ".png", cv::IMREAD_UNCHANGED);

            load_item(item_id, item_icon);
        }
    }
}

void load_hashes_from_file(const string &hash_filename) {
    ifstream dh_stream(hash_filename);
    auto digit_hashes = load_digit_hashes(dh_stream);

    for (size_t i = 0; i < digit_hashes.size(); ++i) {
        load_hash(to_string(i), digit_hashes.at(i));
    }
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cerr << "usage: \n./recognizer <scene_filename...> " << std::endl;
        return 0;
    }

    const auto filenames = std::vector<std::string>(argv + 1, argv + argc);

    vector<cv::Mat> mats;
    mats.reserve(filenames.size());
    for (const auto &filename: filenames) {
        mats.push_back(cv::imread(filename));
    }

    load_items_from_file("items/data/item_table.json",
                         "items/data/item_icon_table.json");
    load_hashes_from_file("fonts/NotoSansSC-Regular_DigitHash.json");

    auto result = recognize(mats);
    auto json_str = export_penguin_arkplanner_json(result);

    cout << json_str << endl;

    return 0;
}