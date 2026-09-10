#include "ocr_engine.hpp"
#include "plate_formatter.hpp"

#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

struct PlateOCR::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "PlateOCR"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<const char*> input_names_c;
    std::vector<const char*> output_names_c;

    // From config
    int img_h = 64;
    int img_w = 128;
    int max_slots = 10;
    std::string alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    char pad_char = '_';
    bool is_rgb = true;

    Impl(const std::string& onnx_path, const std::string& config_path) {
        // ---- Load config (simple parser) ----
        std::ifstream cfg(config_path);
        if (!cfg) throw std::runtime_error("Cannot open plate config: " + config_path);

        std::string line;
        while (std::getline(cfg, line)) {
            if (line.find("max_plate_slots:") != std::string::npos)
                max_slots = std::stoi(line.substr(line.find(':') + 1));
            else if (line.find("img_height:") != std::string::npos)
                img_h = std::stoi(line.substr(line.find(':') + 1));
            else if (line.find("img_width:") != std::string::npos)
                img_w = std::stoi(line.substr(line.find(':') + 1));
            else if (line.find("alphabet:") != std::string::npos) {
                auto start = line.find('\'');
                auto end   = line.rfind('\'');
                if (start != std::string::npos && end > start)
                    alphabet = line.substr(start + 1, end - start - 1);
            }
            else if (line.find("pad_char:") != std::string::npos) {
                auto start = line.find('\'');
                if (start != std::string::npos)
                    pad_char = line[start + 1];
            }
            else if (line.find("image_color_mode:") != std::string::npos)
                is_rgb = line.find("rgb") != std::string::npos;
        }

        // ---- ONNX Runtime ----
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = std::make_unique<Ort::Session>(env, onnx_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Input
        auto in_name = session->GetInputNameAllocated(0, allocator);
        input_name = in_name.get();
        input_names_c = {input_name.c_str()};

        // Outputs (we take the first output that looks like character logits)
        size_t num_outputs = session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(name.get());
        }
        for (auto& n : output_names)
            output_names_c.push_back(n.c_str());

        std::cout << "[OCR] Loaded " << onnx_path
                  << "  input=" << img_w << "x" << img_h
                  << "  slots=" << max_slots
                  << "  alphabet size=" << alphabet.size() << std::endl;
    }

    std::vector<std::pair<std::string, float>> run(const cv::Mat& plate_bgr) {
        if (plate_bgr.empty()) return {};

        // ---- Preprocess ----
        cv::Mat rgb;
        if (is_rgb)
            cv::cvtColor(plate_bgr, rgb, cv::COLOR_BGR2RGB);
        else
            cv::cvtColor(plate_bgr, rgb, cv::COLOR_BGR2GRAY);

        // Keep as uint8, channels-last
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

        // Keep as uint8, channels-last (NHWC) — this is what the official
        // fast-plate-ocr ONNX exports expect.
        if (!resized.isContinuous())
            resized = resized.clone();

        const int channels = is_rgb ? 3 : 1;
        std::vector<uint8_t> input_tensor(
            resized.data,
            resized.data + resized.total() * resized.channels());

        // NHWC
        std::array<int64_t, 4> input_shape{1, img_h, img_w, channels};

        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_ort = Ort::Value::CreateTensor<uint8_t>(
            mem_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size());

        auto outputs = session->Run(Ort::RunOptions{nullptr},
                                    input_names_c.data(), &input_ort, 1,
                                    output_names_c.data(), output_names_c.size());

        // ---- Decode (assume first output is [1, max_slots, vocab] or [1, max_slots*vocab]) ----
        float* data = outputs[0].GetTensorMutableData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        // Handle common shapes
        int slots = max_slots;
        int vocab = static_cast<int>(alphabet.size());

        if (shape.size() == 3) {          // [1, slots, vocab]
            slots = static_cast<int>(shape[1]);
            vocab = static_cast<int>(shape[2]);
        } else if (shape.size() == 2) {   // [1, slots*vocab]
            // fallback
        }

        std::string raw;
        float min_prob = 1.0f;

        for (int s = 0; s < slots; ++s) {
            float* row = data + s * vocab;
            int best = 0;
            float best_p = row[0];
            for (int v = 1; v < vocab; ++v) {
                if (row[v] > best_p) {
                    best_p = row[v];
                    best = v;
                }
            }
            // softmax not always needed – many models already output probabilities
            // but we take the max as confidence
            char ch = (best < static_cast<int>(alphabet.size())) ? alphabet[best] : pad_char;
            if (ch != pad_char) {
                raw += ch;
                min_prob = std::min(min_prob, best_p);
            }
        }

        auto formatted = format_plate_text(raw);
        if (!formatted) return {};

        return {{*formatted, min_prob}};
    }
};

PlateOCR::PlateOCR(const std::string& onnx_path, const std::string& config_path)
    : impl_(std::make_unique<Impl>(onnx_path, config_path)) {}

PlateOCR::~PlateOCR() = default;

std::vector<std::pair<std::string, float>> PlateOCR::run(const cv::Mat& plate_bgr) {
    return impl_->run(plate_bgr);
}