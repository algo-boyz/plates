#pragma once
#include <opencv2/opencv.hpp>
#include <string>

void annotate_plate(cv::Mat& frame, const cv::Rect& box, const std::string& plate_text);