//
// Created by Flandia on 2021/12/11.
//

#include "item.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace nlohmann;
using namespace cv;

std::vector<FullItem> load_full_items(istream &in) {
    json json_obj;

    in >> json_obj;

    vector<FullItem> full_items;
    for (auto &item_json_obj: json_obj["items"]) {
        auto item_id = item_json_obj["itemId"].get<string>();
        auto item_name = item_json_obj["name"].get<string>();
        auto sort_id = item_json_obj["sortId"].get<int>();
        FullItem item{item_id, item_name, sort_id};

        full_items.push_back(item);
    }

    return full_items;
}

std::vector<Item> load_items(istream &in, const vector<FullItem> &full_items) {
    json json_obj;

//    ifstream in("ITEMS/data/item_icon_table.json");
    in >> json_obj;

    vector<Item> items;
    for (auto &item_json_obj: json_obj) {
        auto item_id = item_json_obj["id"].get<string>();
        auto item_name = item_json_obj["name"].get<string>();
        auto sb = ostringstream();
        sb << "items/icon/" << item_id << ".png";

        auto full_item = find_if(
                full_items.begin(), full_items.end(),
                [&item_name](const auto &f) { return f.item_name == item_name; }
        );

        auto item_icon = imread(sb.str(), IMREAD_UNCHANGED);
        Item item{full_item->item_id, item_name, item_icon};

        items.push_back(item);
    }

    return items;
}


std::string FullItem::to_string() const {
    std::ostringstream out;
    out << item_name << "(" << item_id << "; FULL)";
    return out.str();
}

std::string Item::to_string() const {
    std::ostringstream out;
    out << item_name << "(" << item_id << ")";
    return out.str();
}

