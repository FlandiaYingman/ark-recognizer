//
// Created by Flandia on 2021/12/11.
//

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "item.h"
#include "recognize_icon.h"
#include "recognize_type.h"
#include "recognize_number.h"
#include "recognizer.h"

using namespace std;
using namespace cv;
using namespace nlohmann;


static vector<Item> ITEMS;
static map<string, Mat> TEMPL_MAP; // NOLINT(cert-err58-cpp)
static map<string, Mat> MASK_MAP; // NOLINT(cert-err58-cpp)
static map<string, string> HASH_MAP; // NOLINT(cert-err58-cpp)


void load_item(const string &item_id, const Mat &icon) {
    ITEMS.push_back({item_id, "<null>", icon});
    std::tie(TEMPL_MAP[item_id], MASK_MAP[item_id]) = transform_icon_templ(icon);
}

void load_hash(const string &name, const string &hash) {
    HASH_MAP[name] = hash;
}


vector<RecognizeResult> merge_recognize_results(const vector<vector<RecognizeResult>> &recognize_results) {
    auto flatten_recognize_results = flatten(recognize_results);
    auto merged_recognize_results = map<string, RecognizeResult>();

    for (const auto &result: flatten_recognize_results) {
        auto item = result.item;
        if (merged_recognize_results.count(item.item_id)) {
            // item already exists
            auto this_number = result.number;
            auto that_number = merged_recognize_results[item.item_id].number;
            if (this_number != that_number) {
                //TODO: Throw a recognize_all exception.
                printf("merge conflict: item %s with number %d and %d", item.item_id.c_str(), this_number, that_number);
            }
        } else {
            // item not exists
            merged_recognize_results[item.item_id] = result;
        }
    }
    for (const auto &item: ITEMS) {
        if (!merged_recognize_results.count(item.item_id)) {
            merged_recognize_results[item.item_id] = RecognizeResult{item, 0};
        }
    }

    vector<RecognizeResult> result_vector;
    transform(merged_recognize_results.begin(), merged_recognize_results.end(),
              back_inserter(result_vector), [](auto &pair) -> RecognizeResult { return pair.second; });

    return result_vector;
}

vector<RecognizeResult> recognize_one(const Mat &scene_image) {
    auto ir_results = recognize_icon(scene_image);
    auto tr_results = recognize_type(scene_image, ir_results, ITEMS, TEMPL_MAP, MASK_MAP);
    auto nr_results = recognize_number(scene_image, tr_results, HASH_MAP);
    vector<RecognizeResult> scene_results;
    transform(nr_results.begin(), nr_results.end(),
              back_inserter(scene_results),
              [](NRResult &nr_result) { return RecognizeResult{nr_result.item, nr_result.number}; });
    return scene_results;
}

vector<RecognizeResult> recognize(const vector<Mat> &scene_images) {
    vector<vector<RecognizeResult>> recognize_results;
    for (const auto &scene_image: scene_images) {
        auto begin = chrono::steady_clock::now();

        auto scene_results = recognize_one(scene_image);
        recognize_results.push_back(scene_results);

        auto end = chrono::steady_clock::now();
        cout << "used " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;
    }
    vector<RecognizeResult> result = merge_recognize_results(recognize_results);

    return result;
}


string export_penguin_arkplanner_json(const vector<RecognizeResult> &results) {
    json json_obj = json::object();
    json_obj["@type"] = "@penguin-statistics/planner/config";
    json_obj["items"] = json::array();

    for (const auto &item: results) {
        json_obj["items"].push_back(json(item));
    }

    auto json_str = to_string(json_obj);
    return json_str;
}
