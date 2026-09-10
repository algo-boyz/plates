#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include <string>

struct PlateTrack {
    cv::Rect box;
    std::optional<std::string> text;
    float conf = 0.0f;
    int misses = 0;
};

class PlateTracker {
public:
    // Returns the current active tracks after matching
    std::vector<PlateTrack> update(const std::vector<cv::Rect>& detections);

    // Optional: call this later when OCR is ready
    void set_text(size_t track_idx, const std::string& text, float conf);

private:
    std::vector<PlateTrack> tracks_;
    static constexpr float MATCH_IOU = 0.20f;
    static constexpr int   MAX_MISSES = 60;

    static float iou(const cv::Rect& a, const cv::Rect& b);
};