#include "annotator.hpp"
#include <algorithm>

static const int LABEL_BOX_WIDTH  = 180;
static const int LABEL_BOX_HEIGHT = 40;

void annotate_plate(cv::Mat& frame, const cv::Rect& box, const std::string& plate_text) {
    const int x1 = box.x;
    const int y1 = box.y;
    const int x2 = box.x + box.width;
    const int y2 = box.y + box.height;

    // Red detection box
    cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 4);

    const int box_w = LABEL_BOX_WIDTH;
    const int box_h = LABEL_BOX_HEIGHT;
    const int pad = 6;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const int thickness = 2;

    // Find largest font scale that fits
    double font_scale = 0.3;
    for (double scale = 1.2; scale >= 0.3; scale -= 0.05) {
        int baseline = 0;
        cv::Size ts = cv::getTextSize(plate_text, font, scale, thickness, &baseline);
        if (ts.width <= box_w - 2 * pad && ts.height + baseline <= box_h - 2 * pad) {
            font_scale = scale;
            break;
        }
    }

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(plate_text, font, font_scale, thickness, &baseline);

    int center_x = (x1 + x2) / 2;
    int bg_x1 = std::clamp(center_x - box_w / 2, 0, std::max(0, frame.cols - box_w));
    int bg_y1 = (y1 - box_h >= 0) ? y1 - box_h : std::min(y2, frame.rows - box_h);
    bg_y1 = std::max(0, bg_y1);
    int bg_x2 = bg_x1 + box_w;
    int bg_y2 = bg_y1 + box_h;

    // White label background
    cv::rectangle(frame, cv::Point(bg_x1, bg_y1), cv::Point(bg_x2, bg_y2),
                  cv::Scalar(255, 255, 255), -1);

    // Centered text
    int text_x = bg_x1 + (box_w - text_size.width) / 2;
    int text_y = bg_y1 + (box_h + text_size.height) / 2;
    cv::putText(frame, plate_text, cv::Point(text_x, text_y),
                font, font_scale, cv::Scalar(0, 0, 0), thickness);
}