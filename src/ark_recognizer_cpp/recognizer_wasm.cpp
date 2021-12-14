//
// Created by Flandia on 2021/12/14.
//

#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "recognizer.h"
#include "item.h"


extern "C" [[maybe_unused]]
void load_item(const char *item_id, uint8_t *icon_data, int icon_len) {
    std::vector<uint8_t> icon_data_vector(icon_data, icon_data + icon_len);
    auto icon = cv::imdecode(icon_data_vector, cv::IMREAD_UNCHANGED);

    load_item(std::string(item_id), icon);
}

extern "C" [[maybe_unused]]
void load_hash(const char *name, const char *hash) {
    load_hash(std::string(name), std::string(hash));
}


extern "C" [[maybe_unused]]
char *recognize_to_arkplanner_json(const uint8_t **scene_images_data,
                                   const size_t *scene_images_data_length,
                                   size_t scene_image_length) {
    std::vector<cv::Mat> mats;
    for (size_t i = 0; i < scene_image_length; ++i) {
        auto scene_image_data_arr = scene_images_data[i];
        auto scene_image_data_len = scene_images_data_length[i];

        auto scene_image_data = std::vector<uint8_t>(scene_image_data_arr,
                                                     scene_image_data_arr + scene_image_data_len);
        auto scene_image = cv::imdecode(scene_image_data, cv::IMREAD_COLOR);

        mats.push_back(scene_image);
    }

    auto merged_results = recognize(mats);

    auto json_str = export_penguin_arkplanner_json(merged_results);
    auto json_str_len = strlen(json_str.c_str());

    auto json_c_str = new char[json_str_len + 1];
    strcpy(json_c_str, json_str.c_str());

    // note: if calling this function as WASM function, Emscripten free the memory automatically
    return json_c_str;
}

extern "C" int main() {
    using std::cout;
    using std::endl;
    cout << "recognizer_wasm has been successfully loaded." << endl;
    cout << "recognizer_wasm usage: " << endl;
    cout << "\t"
         << "1. call 'void load_item(const char *item_id, uint8_t *icon_data, int icon_len)' with items" << endl;
    cout << "\t"
         << "2. call 'void load_hash(const char *item_id, uint8_t *icon_data, int icon_len)' with hashes" << endl;
    cout << "\tcall char *recognize_to_arkplanner_json(const uint8_t **scene_images_data,\n"
            "\t                                        const size_t *scene_images_data_length,\n"
            "\t                                        size_t scene_image_length) \n"
            "\twith scene images" << endl;
}