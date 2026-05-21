#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace onnx::yolo {
// Ultralytics YOLO detection: x/y are box center coordinates.
struct Detection {
  float x;
  float y;
  float w;
  float h;
  int class_id;
  float confidence;
};
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
  // Always returns batches; a single-instance tensor is batch size 1.
  std::vector<std::vector<Detection>> decode(const Ort::Value &output_tensor,
                                             float conf_threshold = 0.25f);
  std::vector<Detection> nms(const std::vector<Detection> &detections,
                             float iou_threshold = 0.7f);
  std::vector<std::vector<Detection>>
  nms(const std::vector<std::vector<Detection>> &batch_detections,
      float iou_threshold = 0.7f);
};
} // namespace onnx::yolo
