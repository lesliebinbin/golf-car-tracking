#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace onnx::yolo {
class Runner {
private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;
  std::string input_name;
  std::string output_name;
  ONNXTensorElementDataType input_element_type;
  bool dynamic_batch;
  int64_t model_batch;
  int input_channels;
  int input_height;
  int input_width;

public:
  explicit Runner(const char *model_path);
  Ort::Value run(const std::vector<cv::Mat> &input_frames);
  Ort::Value run(const cv::Mat &input_frame);
};
} // namespace onnx::yolo
