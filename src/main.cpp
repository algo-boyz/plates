#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <numeric>
#include <algorithm>

#include "rfdetr_model.hpp"
#include "plate_tracker.hpp"
#include "annotator.hpp"
#include "ocr_engine.hpp"

// Simple greedy NMS (IoU threshold)
static std::vector<cv::Rect> nms(const std::vector<cv::Rect>& boxes,
                                 const std::vector<float>& scores,
                                 float iou_thresh = 0.45f)
{
    if (boxes.empty()) return {};

    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // sort by score descending
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<cv::Rect> keep;

    auto iou = [](const cv::Rect& a, const cv::Rect& b) -> float {
        int ix1 = std::max(a.x, b.x);
        int iy1 = std::max(a.y, b.y);
        int ix2 = std::min(a.x + a.width,  b.x + b.width);
        int iy2 = std::min(a.y + a.height, b.y + b.height);
        int inter = std::max(0, ix2 - ix1) * std::max(0, iy2 - iy1);
        if (inter == 0) return 0.0f;
        float area_a = static_cast<float>(a.area());
        float area_b = static_cast<float>(b.area());
        return inter / (area_a + area_b - inter);
    };

    for (int i : indices) {
        if (suppressed[i]) continue;
        keep.push_back(boxes[i]);
        for (int j : indices) {
            if (i == j || suppressed[j]) continue;
            if (iou(boxes[i], boxes[j]) > iou_thresh)
                suppressed[j] = true;
        }
    }
    return keep;
}

int main() {
    // CONFIG
    const std::string model_path  = "models/rfdetr-custom.onnx";
    const std::string video_path  = "assets/sample.mp4";
    const std::string output_path = "assets/output.mp4";
    const std::string device      = "cpu";
    const float detection_threshold = 0.30f;
    const int max_boxes = 100;
    const int plate_class_id = 0;          // keep 0 for your ONNX export

    std::cout << "Loading RF-DETR model...\n";
    rfdetr::RFDETRModel model(model_path, device);
    std::cout << "Model ready.\n";

    PlateOCR ocr("models/cct_s_v2_global.onnx",
                 "models/cct_s_v2_global_plate_config.yaml");
    const float OCR_CONF_THRESHOLD = 0.95f;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    const int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer(output_path,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(width, height));

    PlateTracker tracker;
    cv::Mat frame;
    int frame_idx = 0;

    std::cout << "Processing...\n";

    while (cap.read(frame)) {
        std::vector<rfdetr::Detection> detections;
        rfdetr::Timings timings;

        model.predict(frame, detections, timings, detection_threshold, max_boxes);

        // Collect plate boxes + scores
        std::vector<cv::Rect> plate_boxes;
        std::vector<float>    plate_scores;

        for (const auto& d : detections) {
            // optional debug:
            // std::cout << "label=" << d.label << " score=" << d.score << "\n";

            if (d.label == plate_class_id) {
                const auto& b = d.unnormalizedBox;
                cv::Rect box(
                    static_cast<int>(b.x),
                    static_cast<int>(b.y),
                    static_cast<int>(b.width),
                    static_cast<int>(b.height)
                );
                // clamp
                box &= cv::Rect(0, 0, frame.cols, frame.rows);
                if (box.width > 2 && box.height > 2) {
                    plate_boxes.push_back(box);
                    plate_scores.push_back(d.score);
                }
            }
        }

        // NMS – removes overlapping duplicates that create ghost tracks
        auto clean_boxes = nms(plate_boxes, plate_scores, 0.45f);

        // Update tracker with the cleaned set
        auto tracks = tracker.update(clean_boxes);

        // OCR + lock text (only once per track)
        for (size_t i = 0; i < tracks.size(); ++i) {
            auto& t = tracks[i];

            if (!t.text.has_value()) {
                // safety: empty crop
                if (t.box.width <= 2 || t.box.height <= 2) continue;

                cv::Mat crop = frame(t.box);
                auto results = ocr.run(crop);
                if (!results.empty()) {
                    auto [text, conf] = results[0];
                    if (conf >= OCR_CONF_THRESHOLD) {
                        tracker.set_text(i, text, conf);
                        t.text = text;          // for drawing this frame
                    }
                }
            }
        }

        // Draw ONLY tracks that were matched this frame (misses == 0)
        // → eliminates the lingering “ghost” boxes
        for (const auto& t : tracks) {
            if (t.misses > 0) continue;     // skip aged-out / unmatched tracks

            if (t.text.has_value()) {
                annotate_plate(frame, t.box, *t.text);
            } else {
                cv::rectangle(frame, t.box, cv::Scalar(0, 0, 255), 3);
            }
        }

        writer.write(frame);

        if (++frame_idx % 30 == 0) {
            std::cout << "Frame " << frame_idx
                      << " | infer " << timings.ort_run << " ms"
                      << " | raw plates " << plate_boxes.size()
                      << " | after NMS " << clean_boxes.size() << "\n";
        }

        cv::imshow("Vehicle Plates", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    std::cout << "Done → " << output_path << std::endl;
    return 0;
}