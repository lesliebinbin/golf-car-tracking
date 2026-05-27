#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <onnx.hpp>
#include <opencv2/opencv.hpp>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

#include "video_processing.hpp"

namespace {

constexpr int kYoloImageSize = 640;
constexpr int kYoloOutputValues = 5;
constexpr float kConfThreshold = 0.25f;
constexpr float kIouThreshold = 0.7f;

std::filesystem::path current_source_dir(
    const std::source_location location = std::source_location::current()) {
  std::filesystem::path source_path{location.file_name()};
  if (source_path.is_relative()) {
    const std::filesystem::path executable_dir =
        std::filesystem::read_symlink("/proc/self/exe").parent_path();
    source_path = executable_dir / source_path;
  }

  return std::filesystem::weakly_canonical(source_path).parent_path();
}

std::string project_file(const std::filesystem::path &relative_path) {
  return (current_source_dir() / ".." / ".." / relative_path)
      .lexically_normal()
      .string();
}

cv::Mat read_image(const std::string &path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + path);
  }
  return image;
}

cv::Mat make_model_ready_frame(const std::string &path) {
  video_processing::ImageHandler image_handler;
  video_processing::LetterBoxResult letterbox =
      image_handler.letterbox(read_image(path), cv::Size(kYoloImageSize, kYoloImageSize));

  cv::Mat rgb;
  cv::cvtColor(letterbox.image, rgb, cv::COLOR_BGR2RGB);

  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
  return normalized;
}

void assert_yolo_output(const Ort::Value &output, int64_t expected_batch,
                        const std::string &label) {
  if (!output.IsTensor()) {
    throw std::runtime_error(label + " output is not a tensor");
  }

  Ort::TensorTypeAndShapeInfo info = output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();
  if (shape.size() != 3) {
    throw std::runtime_error(label + " output must have rank 3");
  }

  if (shape[0] != expected_batch) {
    throw std::runtime_error(label + " output batch dimension mismatch");
  }

  const bool channels_first = shape[1] == kYoloOutputValues;
  const bool channels_last = shape[2] == kYoloOutputValues;
  if (!channels_first && !channels_last) {
    throw std::runtime_error(label + " output does not look like YOLO detect output");
  }

  const float *data = output.GetTensorData<float>();
  const int64_t anchors = channels_first ? shape[2] : shape[1];
  float max_score = 0.0f;
  int confident_count = 0;

  for (int64_t batch = 0; batch < expected_batch; ++batch) {
    for (int64_t anchor = 0; anchor < anchors; ++anchor) {
      const int64_t index =
          channels_first
              ? batch * shape[1] * shape[2] + 4 * shape[2] + anchor
              : batch * shape[1] * shape[2] + anchor * shape[2] + 4;
      const float score = data[index];
      if (!std::isfinite(score)) {
        throw std::runtime_error(label + " output contains non-finite score");
      }
      max_score = std::max(max_score, score);
      if (score >= 0.25f) {
        ++confident_count;
      }
    }
  }

  if (max_score < 0.25f) {
    throw std::runtime_error(label + " output did not produce confident detections");
  }

  std::cout << label << " output shape=(" << shape[0] << ", " << shape[1]
            << ", " << shape[2] << "), max_score=" << max_score
            << ", scores>=0.25=" << confident_count << std::endl;
}

void assert_decoded_detections(
    const std::vector<onnx::yolo::Detection> &detections,
    const std::string &label) {
  if (detections.empty()) {
    throw std::runtime_error(label + " produced no decoded detections");
  }

  const bool all_valid = std::ranges::all_of(
      detections, [](const onnx::yolo::Detection &detection) {
        const std::array values{detection.x, detection.y, detection.w,
                                detection.h, detection.confidence};
        return std::ranges::all_of(values, [](float value) {
                 return std::isfinite(value);
               }) &&
               detection.w > 0.0f && detection.h > 0.0f &&
               detection.confidence >= kConfThreshold &&
               detection.class_id >= 0;
      });

  if (!all_valid) {
    throw std::runtime_error(label + " decoded detections are invalid");
  }

  const auto best = std::ranges::max_element(
      detections, {}, &onnx::yolo::Detection::confidence);
  std::cout << label << " decoded detections=" << detections.size()
            << ", max_confidence=" << best->confidence << std::endl;
}

void assert_nms_detections(
    const std::vector<onnx::yolo::Detection> &decoded,
    const std::vector<onnx::yolo::Detection> &nms_detections,
    const std::string &label) {
  if (nms_detections.empty()) {
    throw std::runtime_error(label + " NMS produced no detections");
  }

  if (nms_detections.size() > decoded.size()) {
    throw std::runtime_error(label + " NMS increased detection count");
  }

  const bool sorted_by_confidence = std::ranges::is_sorted(
      nms_detections, [](const onnx::yolo::Detection &a,
                         const onnx::yolo::Detection &b) {
        return a.confidence > b.confidence;
      });
  if (!sorted_by_confidence) {
    throw std::runtime_error(label + " NMS detections are not confidence-sorted");
  }

  std::cout << label << " NMS detections=" << nms_detections.size()
            << " from decoded=" << decoded.size() << std::endl;
}

void assert_decoded_batches(
    const std::vector<std::vector<onnx::yolo::Detection>> &batch_detections,
    std::size_t expected_batch_size, const std::string &label) {
  if (batch_detections.size() != expected_batch_size) {
    throw std::runtime_error(label + " decoded batch size mismatch");
  }

  for (std::size_t batch = 0; batch < batch_detections.size(); ++batch) {
    assert_decoded_detections(
        batch_detections[batch],
        label + " batch[" + std::to_string(batch) + "]");
  }
}

void assert_nms_batches(
    const std::vector<std::vector<onnx::yolo::Detection>> &decoded,
    const std::vector<std::vector<onnx::yolo::Detection>> &nms_detections,
    const std::string &label) {
  if (nms_detections.size() != decoded.size()) {
    throw std::runtime_error(label + " NMS batch size mismatch");
  }

  for (std::size_t batch = 0; batch < decoded.size(); ++batch) {
    assert_nms_detections(decoded[batch], nms_detections[batch],
                          label + " batch[" + std::to_string(batch) + "]");
  }
}

void assert_synthetic_nms(onnx::yolo::Runner &runner) {
  const std::vector<onnx::yolo::Detection> synthetic{
      {.x = 100.0f,
       .y = 100.0f,
       .w = 50.0f,
       .h = 50.0f,
       .class_id = 0,
       .confidence = 0.90f},
      {.x = 102.0f,
       .y = 102.0f,
       .w = 50.0f,
       .h = 50.0f,
       .class_id = 0,
       .confidence = 0.80f},
      {.x = 102.0f,
       .y = 102.0f,
       .w = 50.0f,
       .h = 50.0f,
       .class_id = 1,
       .confidence = 0.70f},
      {.x = 300.0f,
       .y = 300.0f,
       .w = 20.0f,
       .h = 20.0f,
       .class_id = 0,
       .confidence = 0.60f},
  };

  const std::vector<onnx::yolo::Detection> kept =
      runner.nms(synthetic, 0.5f);
  if (kept.size() != 3) {
    throw std::runtime_error("synthetic NMS should keep 3 detections");
  }

  const bool suppressed_lower_duplicate = std::ranges::none_of(
      kept, [](const onnx::yolo::Detection &detection) {
        return detection.class_id == 0 &&
               std::abs(detection.confidence - 0.80f) < 1e-6f;
      });
  if (!suppressed_lower_duplicate) {
    throw std::runtime_error("synthetic NMS kept lower-confidence duplicate");
  }

  std::cout << "synthetic NMS passed" << std::endl;
}

} // namespace

int main() {
  try {
    const std::vector<std::string> image_paths = {
        project_file("python/video_frames/frame_0140.jpg"),
        project_file("python/video_frames/frame_0139.jpg"),
        project_file("python/video_frames/frame_0148.jpg"),
    };

    std::vector<cv::Mat> frames;
    frames.reserve(image_paths.size());
    for (const std::string &path : image_paths) {
      frames.push_back(make_model_ready_frame(path));
    }

    onnx::yolo::Runner static_runner(
        project_file("python/golf-car-static.onnx").c_str());
    Ort::Value static_output = static_runner.run(frames.front());
    assert_yolo_output(static_output, 1, "static model");
    const std::vector<std::vector<onnx::yolo::Detection>> static_decoded =
        static_runner.decode(static_output, kConfThreshold);
    assert_decoded_batches(static_decoded, 1, "static model");
    const std::vector<std::vector<onnx::yolo::Detection>> static_nms =
        static_runner.nms(static_decoded, kIouThreshold);
    assert_nms_batches(static_decoded, static_nms, "static model");
    assert_synthetic_nms(static_runner);

    onnx::yolo::Runner dynamic_runner(
        project_file("python/golf-car-dynamic.onnx").c_str());
    Ort::Value dynamic_output = dynamic_runner.run(frames);
    assert_yolo_output(dynamic_output, static_cast<int64_t>(frames.size()),
                       "dynamic model");
    const std::vector<std::vector<onnx::yolo::Detection>> dynamic_decoded =
        dynamic_runner.decode(dynamic_output, kConfThreshold);
    assert_decoded_batches(dynamic_decoded, frames.size(), "dynamic model");
    const std::vector<std::vector<onnx::yolo::Detection>> dynamic_nms =
        dynamic_runner.nms(dynamic_decoded, kIouThreshold);
    assert_nms_batches(dynamic_decoded, dynamic_nms, "dynamic model");

    std::cout << "YOLO ONNX inference test passed" << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "YOLO ONNX inference test failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}