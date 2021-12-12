//
// Created by Flandia on 2021/12/11.
//

#include "item.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include "recognize_icon.h"
#include "recognize_type.h"
#include "recognize_number.h"


using namespace std;
using namespace cv;


void draw_ir_results(Mat &scene_canvas, const vector<IRResult> &ir_results) {
    for (const auto &result: ir_results) {
        int x = (int) round(result.x), y = (int) round(result.y), r = (int) round(result.radius);
        circle(scene_canvas, Point(x, y), r, Scalar(0, 0, 255));
        circle(scene_canvas, Point(x, y), 1, Scalar(0, 255, 255));
    }
}

void draw_tr_results(Mat &scene_canvas, const vector<TRResult> &tr_results) {
    for (const auto &tr_result: tr_results) {
        rectangle(scene_canvas,
                  tr_result.loc,
                  Size(tr_result.loc) + tr_result.size,
                  Scalar(255.0, 0.0, 0.0));
        putText(scene_canvas,
                tr_result.item.item_id,
                tr_result.loc,
                FONT_HERSHEY_DUPLEX,
                1.0,
                Scalar(255.0, 0.0, 0.0));
    }
}

void draw_nr_results(Mat &scene_canvas, const vector<NRResult> &nr_results) {
    for (const auto &nr_result: nr_results) {
        rectangle(scene_canvas, Rect(nr_result.loc, nr_result.size), Scalar(0, 0, 255));
        putText(scene_canvas, to_string(nr_result.number), nr_result.loc, FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255));
    }
}


class RecognizeResult {
public:
    Item item;
    int number = -1;
};


vector<RecognizeResult> recognize(const vector<string> &filenames);

static vector<Item> ITEMS;
static map<string, Mat> TEMPL_MAP; // NOLINT(cert-err58-cpp)
static map<string, Mat> MASK_MAP; // NOLINT(cert-err58-cpp)
static map<string, string> HASH_MAP; // NOLINT(cert-err58-cpp)


extern "C" void load_item(const char *item_id, uint8_t *icon_data, int icon_len) {
    vector<uint8_t> icon_data_vector(icon_data, icon_data + icon_len);
    auto icon = imdecode(icon_data_vector, IMREAD_UNCHANGED);
    ITEMS.push_back({item_id, "<null>", icon});
    tie(TEMPL_MAP[item_id], MASK_MAP[item_id]) = transform_icon_templ(icon);
}

extern "C" void load_hash(const char *name, const char *hash) {
    HASH_MAP[name] = hash;
}


void load_items_from_file() {
    static auto ITEM_TABLE = "ITEMS/data/item_table.json";
    static auto ITEM_ICON_TABLE = "ITEMS/data/item_icon_table.json";

    ifstream it_stream(ITEM_TABLE);
    auto full_items = load_full_items(it_stream);
    ifstream iit_stream(ITEM_ICON_TABLE);
    auto items = load_items(iit_stream, full_items);

    for (const auto &item: items) {
        auto item_id = item.item_id.c_str();
        vector<uint8_t> buf;
        imencode(".png", item.item_icon, buf);
        load_item(item_id, buf.data(), (int) buf.size());
    }
}

void load_hashes_from_file() {
    static auto NOTO_SANS_HASH_JSON = "fonts/NotoSansSC-Regular_DigitHash.json";

    ifstream dh_stream(NOTO_SANS_HASH_JSON);
    auto digit_hashes = load_digit_hashes(dh_stream);

    for (size_t i = 0; i < digit_hashes.size(); ++i) {
        load_hash(to_string(i).c_str(), digit_hashes.at(i).c_str());
    }
}


vector<RecognizeResult> merge_to_recognize_result(vector<vector<RecognizeResult>> recognize_results) {
    auto flatten_recognize_results = flatten(recognize_results);
    auto merged_recognize_results = map<string, RecognizeResult>();

    for (const auto &result: flatten_recognize_results) {
        auto item = result.item;
        if (merged_recognize_results.count(item.item_id)) {
            // item already exists
            auto this_number = result.number;
            auto that_number = merged_recognize_results[item.item_id].number;
            if (this_number != that_number) {
                //TODO: Throw a recognize exception.
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

vector<RecognizeResult> recognize(const vector<string> &filenames) {
    vector<vector<RecognizeResult>> recognize_results;
    for (const auto &filename: filenames) {
        auto begin = chrono::steady_clock::now();

        auto scene_image = imread(filename);
        auto ir_results = recognize_icon(scene_image);
        auto tr_results = recognize_type(scene_image, ir_results, ITEMS, TEMPL_MAP, MASK_MAP);
        auto nr_results = recognize_number(scene_image, tr_results, HASH_MAP);

        vector<RecognizeResult> scene_results;
        transform(nr_results.begin(), nr_results.end(), back_inserter(scene_results),
                  [](NRResult &nr_result) { return RecognizeResult{nr_result.item, nr_result.number}; });
        recognize_results.push_back(scene_results);

        auto end = chrono::steady_clock::now();
        cout << "used " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;

//        auto scene_canvas = scene_image.clone();
//        draw_ir_results(scene_canvas, ir_results);
//        imshow("ir_results", scene_canvas);
//        draw_tr_results(scene_canvas, tr_results);
//        imshow("tr_results", scene_canvas);
//        draw_nr_results(scene_canvas, nr_results);
//        imshow("nr_results", scene_canvas);

        waitKey();
    }
    vector<RecognizeResult> result = merge_to_recognize_result(recognize_results);

    return result;
}


int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cerr << "usage: \n./recognizer <scene_filename...> " << std::endl;
        return 0;
    }

    const auto filenames = std::vector<std::string>(argv + 1, argv + argc);

    load_items_from_file();
    load_hashes_from_file();

    recognize(filenames);

    return 0;
}
