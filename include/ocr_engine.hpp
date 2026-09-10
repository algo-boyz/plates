#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>
#include <memory>

class PlateOCR {
public:
    PlateOCR(const std::string& onnx_path, const std::string& config_path);
    ~PlateOCR();

    // Returns list of (formatted_text, confidence) sorted by confidence desc
    // Only includes results that pass Turkish format_plate_text()
    std::vector<std::pair<std::string, float>> run(const cv::Mat& plate_bgr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};