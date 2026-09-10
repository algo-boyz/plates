#include "plate_tracker.hpp"
#include <algorithm>

float PlateTracker::iou(const cv::Rect& a, const cv::Rect& b) {
    int ix1 = std::max(a.x, b.x);
    int iy1 = std::max(a.y, b.y);
    int ix2 = std::min(a.x + a.width,  b.x + b.width);
    int iy2 = std::min(a.y + a.height, b.y + b.height);
    int inter = std::max(0, ix2 - ix1) * std::max(0, iy2 - iy1);
    if (inter == 0) return 0.0f;
    float area_a = static_cast<float>(a.area());
    float area_b = static_cast<float>(b.area());
    return inter / (area_a + area_b - inter);
}

std::vector<PlateTrack> PlateTracker::update(const std::vector<cv::Rect>& detections) {
    std::vector<bool> used(tracks_.size(), false);

    for (const auto& det : detections) {
        float best_iou = 0.0f;
        int best_idx = -1;

        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (used[i]) continue;
            float score = iou(det, tracks_[i].box);
            if (score > best_iou) {
                best_iou = score;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx != -1 && best_iou >= MATCH_IOU) {
            // Matched existing track
            tracks_[best_idx].box = det;
            tracks_[best_idx].misses = 0;
            used[best_idx] = true;
        } else {
            // New track
            tracks_.push_back({det, std::nullopt, 0.0f, 0});
            used.push_back(true);
        }
    }

    // Age unmatched tracks
    std::vector<PlateTrack> surviving;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!used[i]) tracks_[i].misses++;
        if (tracks_[i].misses <= MAX_MISSES)
            surviving.push_back(tracks_[i]);
    }
    tracks_ = std::move(surviving);

    return tracks_;
}

void PlateTracker::set_text(size_t track_idx, const std::string& text, float conf) {
    if (track_idx < tracks_.size()) {
        tracks_[track_idx].text = text;
        tracks_[track_idx].conf = conf;
    }
}