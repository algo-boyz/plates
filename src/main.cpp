#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <optional>

#include "rfdetr_model.hpp"
#include "plate_tracker.hpp"
#include "annotator.hpp"
#include "ocr_engine.hpp"

int main() {
    // CONFIG
    const std::string model_path  = "models/checkpoint_best_total.pth";
    const std::string video_path  = "assets/sample.mp4";
    const std::string output_path = "assets/output.mp4";
    const std::string device      = "cpu";
    const float detection_threshold = 0.30f;
    const int max_boxes = 100;
    const int plate_class_id = 0;

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

        // Collect plate boxes
        std::vector<cv::Rect> plate_boxes;
        for (const auto& d : detections) {
            std::cout << "label=" << d.label << " score=" << d.score << "\n";
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
                if (box.width > 2 && box.height > 2)
                    plate_boxes.push_back(box);
            }
        }

        // Update tracker
        auto tracks = tracker.update(plate_boxes);
        for (size_t i = 0; i < tracks.size(); ++i) {
            auto& t = tracks[i];

            // Run OCR only if we don't have a locked high-confidence text yet
            if (!t.text.has_value()) {
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

            if (t.text.has_value()) {
                annotate_plate(frame, t.box, *t.text);
            } else {
                cv::rectangle(frame, t.box, cv::Scalar(0, 0, 255), 3);
            }
        }

        // Draw
        for (const auto& t : tracks) {
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
                      << " | plates " << plate_boxes.size() << "\n";
        }

        cv::imshow("Vehicle Plates", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    std::cout << "Done → " << output_path << std::endl;
    return 0;
}