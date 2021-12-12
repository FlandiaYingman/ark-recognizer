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

std::array<hash_t, 10> load_digit_hashes_from_file() {
    std::ifstream in("fonts/NotoSansSC-Regular_DigitHash.json");
    auto digit_hashes = load_digit_hashes(in);
    return digit_hashes;
}


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

    const auto item_templates = preprocess_item_templates(items);
    const auto item_template_masks = preprocess_item_template_masks(items);
    const auto digit_hashes = load_digit_hashes_from_file();

    for (const auto &item: filenames) {
        auto begin = chrono::steady_clock::now();

        auto scene_image = cv::imread(item);
        auto ir_results = recognize_icon(scene_image);
        auto tr_results = recognize_type(scene_image, ir_results, items, item_templates, item_template_masks);
        auto nr_results = recognize_number(scene_image, tr_results, digit_hashes);

        auto end = chrono::steady_clock::now();

        cout << "used " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;

        auto scene_canvas = scene_image.clone();
        draw_ir_results(scene_canvas, ir_results);
        imshow("ir_results", scene_canvas);

        draw_tr_results(scene_canvas, tr_results);
        imshow("tr_results", scene_canvas);

        draw_nr_results(scene_canvas, nr_results);
        imshow("nr_results", scene_canvas);

        waitKey();
    }

    return 0;
}
