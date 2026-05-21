#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <iostream>
#include <onnx.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "video_processing.hpp"

namespace {

constexpr int kDefaultModelImageSize = 640;
constexpr const char *kWindowName = "Golf Car Movement Tracking";
constexpr double kSeekSeconds = 1.0;

const std::vector<int> kLeftKeys{2, 81, 63234, 65361, 2424832};
const std::vector<int> kRightKeys{3, 83, 63235, 65363, 2555904};

enum class DecayStrategy { SimpleAverage };

struct TrackerConfig {
  std::filesystem::path model_path;
  std::filesystem::path video_path;
  float conf_threshold = 0.25f;
  float iou_threshold = 0.7f;
  int canvas_size = 180;
  int canvas_margin = 16;
  int buffer_size = 20;
  DecayStrategy decay_strategy = DecayStrategy::SimpleAverage;
  std::filesystem::path output_video_path;
  int frame_stride = 1;
  int model_image_size = kDefaultModelImageSize;
};

struct FrameMetadata {
  int width;
  int height;
  double fps;
  int total_frames;
};

struct TrackRecord {
  int frame_index;
  double timestamp;
  float center_x;
  float center_y;
  float confidence;
  cv::Rect2f bbox;
};

bool contains_key(const std::vector<int> &keys, int key) {
  return std::ranges::find(keys, key) != keys.end();
}

template <typename T>
T yaml_value(const YAML::Node &node, const char *key, T default_value) {
  return node[key] ? node[key].as<T>() : default_value;
}

std::filesystem::path resolve_config_path(int argc, char **argv) {
  if (argc > 2) {
    throw std::invalid_argument("Usage: movement_track [tracker_config.yml]");
  }

  if (argc == 2) {
    return std::filesystem::path{argv[1]};
  }

  return std::filesystem::path{"tracker_config.yml"};
}

std::filesystem::path resolve_path(const YAML::Node &node, const char *key,
                                   const std::filesystem::path &base_dir,
                                   bool required) {
  if (!node[key]) {
    if (required) {
      throw std::runtime_error(std::string{"Missing required YAML key: "} +
                               key);
    }
    return {};
  }

  const std::string raw_path = node[key].as<std::string>();
  if (raw_path.empty()) {
    return {};
  }

  std::filesystem::path path{raw_path};
  if (path.is_relative()) {
    path = base_dir / path;
  }
  return path.lexically_normal();
}

DecayStrategy parse_decay_strategy(const YAML::Node &node) {
  const std::string value =
      yaml_value<std::string>(node, "decay_strategy", "simple_average");
  if (value == "simple_average") {
    return DecayStrategy::SimpleAverage;
  }
  throw std::runtime_error("Unsupported decay_strategy: " + value);
}

TrackerConfig load_config(const std::filesystem::path &config_path) {
  if (!std::filesystem::is_regular_file(config_path)) {
    throw std::runtime_error("Config file does not exist: " +
                             config_path.string());
  }

  const YAML::Node node = YAML::LoadFile(config_path.string());
  const std::filesystem::path base_dir = config_path.has_parent_path()
                                             ? config_path.parent_path()
                                             : std::filesystem::current_path();

  TrackerConfig config{
      .model_path = resolve_path(node, "model_path", base_dir, true),
      .video_path = resolve_path(node, "video_path", base_dir, true),
      .conf_threshold = yaml_value<float>(node, "conf_threshold", 0.25f),
      .iou_threshold = yaml_value<float>(node, "iou_threshold", 0.7f),
      .canvas_size = yaml_value<int>(node, "canvas_size", 180),
      .canvas_margin = yaml_value<int>(node, "canvas_margin", 16),
      .buffer_size = yaml_value<int>(node, "buffer_size", 20),
      .decay_strategy = parse_decay_strategy(node),
      .output_video_path = resolve_path(node, "output_path", base_dir, false),
      .frame_stride = yaml_value<int>(node, "frame_stride", 1),
      .model_image_size =
          yaml_value<int>(node, "model_image_size", kDefaultModelImageSize),
  };

  if (!std::filesystem::is_regular_file(config.model_path)) {
    throw std::runtime_error("model_path does not exist: " +
                             config.model_path.string());
  }
  if (!std::filesystem::is_regular_file(config.video_path)) {
    throw std::runtime_error("video_path does not exist: " +
                             config.video_path.string());
  }
  if (config.conf_threshold < 0.0f || config.conf_threshold > 1.0f) {
    throw std::runtime_error("conf_threshold must be in [0, 1]");
  }
  if (config.iou_threshold < 0.0f || config.iou_threshold > 1.0f) {
    throw std::runtime_error("iou_threshold must be in [0, 1]");
  }
  if (config.canvas_size <= 0) {
    throw std::runtime_error("canvas_size must be positive");
  }
  if (config.canvas_margin < 0) {
    throw std::runtime_error("canvas_margin must be non-negative");
  }
  if (config.buffer_size <= 0) {
    throw std::runtime_error("buffer_size must be positive");
  }
  if (config.frame_stride <= 0) {
    throw std::runtime_error("frame_stride must be positive");
  }
  if (config.model_image_size <= 0) {
    throw std::runtime_error("model_image_size must be positive");
  }

  return config;
}

FrameMetadata read_metadata(cv::VideoCapture &capture,
                            const std::filesystem::path &video_path) {
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = capture.get(cv::CAP_PROP_FPS);
  const int total_frames =
      std::max(0, static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT)));

  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Video has invalid dimensions: " +
                             video_path.string());
  }
  if (!std::isfinite(fps) || fps <= 0.0 || fps > 240.0) {
    fps = 30.0;
  }

  return {.width = width,
          .height = height,
          .fps = fps,
          .total_frames = total_frames};
}

cv::VideoWriter create_writer(const TrackerConfig &config,
                              const FrameMetadata &metadata) {
  if (config.output_video_path.empty()) {
    return {};
  }

  if (config.output_video_path.has_parent_path()) {
    std::filesystem::create_directories(config.output_video_path.parent_path());
  }

  cv::VideoWriter writer{config.output_video_path.string(),
                         cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                         metadata.fps,
                         cv::Size{metadata.width, metadata.height}};
  if (!writer.isOpened()) {
    throw std::runtime_error("Failed to open output video writer: " +
                             config.output_video_path.string());
  }

  return writer;
}

cv::Mat make_model_ready_frame(const cv::Mat &frame,
                               video_processing::ImageHandler &image_handler,
                               int model_image_size,
                               video_processing::LetterBoxResult &letterbox) {
  letterbox = image_handler.letterbox(
      frame, cv::Size{model_image_size, model_image_size});

  cv::Mat rgb;
  cv::cvtColor(letterbox.image, rgb, cv::COLOR_BGR2RGB);

  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
  return normalized;
}

float clamp_float(float value, float lower, float upper) {
  return std::min(std::max(value, lower), upper);
}

TrackRecord
detection_to_track_record(const onnx::yolo::Detection &detection,
                          const video_processing::LetterBoxResult &letterbox,
                          const FrameMetadata &metadata, int frame_index,
                          double timestamp) {
  float center_x =
      (detection.x - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
  float center_y =
      (detection.y - static_cast<float>(letterbox.pad_y)) / letterbox.scale;
  float width = detection.w / letterbox.scale;
  float height = detection.h / letterbox.scale;

  center_x =
      clamp_float(center_x, 0.0f, static_cast<float>(metadata.width - 1));
  center_y =
      clamp_float(center_y, 0.0f, static_cast<float>(metadata.height - 1));
  width = clamp_float(width, 0.0f, static_cast<float>(metadata.width));
  height = clamp_float(height, 0.0f, static_cast<float>(metadata.height));

  return {.frame_index = frame_index,
          .timestamp = timestamp,
          .center_x = center_x,
          .center_y = center_y,
          .confidence = detection.confidence,
          .bbox = cv::Rect2f{center_x - width / 2.0f, center_y - height / 2.0f,
                             width, height}};
}

void draw_detection(cv::Mat &frame, const TrackRecord &record) {
  const int x1 = static_cast<int>(
      std::round(clamp_float(record.bbox.x, 0.0f, frame.cols - 1.0f)));
  const int y1 = static_cast<int>(
      std::round(clamp_float(record.bbox.y, 0.0f, frame.rows - 1.0f)));
  const int x2 = static_cast<int>(std::round(
      clamp_float(record.bbox.x + record.bbox.width, 0.0f, frame.cols - 1.0f)));
  const int y2 = static_cast<int>(std::round(clamp_float(
      record.bbox.y + record.bbox.height, 0.0f, frame.rows - 1.0f)));
  const cv::Point center{static_cast<int>(std::round(record.center_x)),
                         static_cast<int>(std::round(record.center_y))};

  cv::rectangle(frame, cv::Point{x1, y1}, cv::Point{x2, y2},
                cv::Scalar{0, 255, 0}, 2);
  cv::circle(frame, center, 4, cv::Scalar{0, 0, 255}, -1);
  cv::putText(frame, std::format("Golf Car {:.2f}", record.confidence),
              cv::Point{x1, std::max(20, y1 - 8)}, cv::FONT_HERSHEY_SIMPLEX,
              0.6, cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
}

cv::Point map_to_canvas(const TrackRecord &record,
                        const FrameMetadata &metadata, int canvas_size) {
  const auto x = static_cast<int>(std::round(
      record.center_x / static_cast<float>(std::max(1, metadata.width - 1)) *
      static_cast<float>(canvas_size - 1)));
  const auto y = static_cast<int>(std::round(
      record.center_y / static_cast<float>(std::max(1, metadata.height - 1)) *
      static_cast<float>(canvas_size - 1)));
  return {std::clamp(x, 0, canvas_size - 1), std::clamp(y, 0, canvas_size - 1)};
}

cv::Mat create_tracker_canvas(const std::vector<TrackRecord> &current_records,
                              const FrameMetadata &metadata, int size) {
  cv::Mat canvas{size, size, CV_8UC3, cv::Scalar{24, 24, 24}};
  cv::rectangle(canvas, cv::Point{0, 0}, cv::Point{size - 1, size - 1},
                cv::Scalar{180, 180, 180}, 1);
  cv::putText(canvas, "Track", cv::Point{8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar{220, 220, 220}, 1, cv::LINE_AA);

  for (const TrackRecord &record : current_records) {
    const cv::Point point = map_to_canvas(record, metadata, size);
    cv::circle(canvas, point, 5, cv::Scalar{0, 0, 255}, -1);
    cv::circle(canvas, point, 7, cv::Scalar{255, 255, 255}, 1);
  }

  return canvas;
}

cv::Mat simple_average(const std::deque<cv::Mat> &buffer) {
  if (buffer.empty()) {
    throw std::invalid_argument("tracker overlay buffer must not be empty");
  }

  cv::Mat accumulator = cv::Mat::zeros(buffer.front().size(), CV_32FC3);
  for (const cv::Mat &canvas : buffer) {
    cv::Mat float_canvas;
    canvas.convertTo(float_canvas, CV_32FC3);
    accumulator += float_canvas;
  }

  accumulator /= static_cast<double>(buffer.size());

  cv::Mat result;
  accumulator.convertTo(result, CV_8UC3);
  return result;
}

void draw_tracking_canvas(cv::Mat &frame,
                          const std::vector<TrackRecord> &current_records,
                          const FrameMetadata &metadata,
                          const TrackerConfig &config,
                          std::deque<cv::Mat> &overlay_buffer) {
  const int size =
      std::min({config.canvas_size, frame.cols - 2 * config.canvas_margin,
                frame.rows - 2 * config.canvas_margin});
  if (size <= 20) {
    return;
  }

  cv::Mat canvas = create_tracker_canvas(current_records, metadata, size);
  if (!overlay_buffer.empty() &&
      overlay_buffer.front().size() != canvas.size()) {
    overlay_buffer.clear();
  }

  overlay_buffer.push_back(canvas);
  while (static_cast<int>(overlay_buffer.size()) > config.buffer_size) {
    overlay_buffer.pop_front();
  }

  cv::Mat blended_canvas;
  switch (config.decay_strategy) {
  case DecayStrategy::SimpleAverage:
    blended_canvas = simple_average(overlay_buffer);
    break;
  }

  const int x1 = frame.cols - config.canvas_margin - size;
  const int y1 = frame.rows - config.canvas_margin - size;
  cv::Mat roi = frame(cv::Rect{x1, y1, size, size});
  cv::Mat blended;
  cv::addWeighted(roi, 0.35, blended_canvas, 0.65, 0.0, blended);
  blended.copyTo(roi);
}

cv::Mat annotate_frame(const cv::Mat &frame,
                       const std::vector<TrackRecord> &current_records,
                       const FrameMetadata &metadata,
                       const TrackerConfig &config,
                       std::deque<cv::Mat> &overlay_buffer) {
  cv::Mat annotated = frame.clone();
  for (const TrackRecord &record : current_records) {
    draw_detection(annotated, record);
  }
  draw_tracking_canvas(annotated, current_records, metadata, config,
                       overlay_buffer);
  return annotated;
}

double frame_timestamp(const cv::VideoCapture &capture, int frame_index,
                       double fps) {
  const double timestamp_ms = capture.get(cv::CAP_PROP_POS_MSEC);
  if (std::isfinite(timestamp_ms) && timestamp_ms > 0.0) {
    return timestamp_ms / 1000.0;
  }
  return static_cast<double>(frame_index) / fps;
}

struct PlayerAction {
  bool paused;
  bool should_quit;
  int seek_delta;
};

PlayerAction wait_for_player(int delay_ms, bool paused, double fps) {
  const int seek_frames =
      std::max(1, static_cast<int>(std::round(fps * kSeekSeconds)));

  while (true) {
    const int key = cv::waitKeyEx(paused ? 0 : delay_ms);
    const int key_ascii = key & 0xFF;

    if (key_ascii == 27 || key_ascii == 'q') {
      return {.paused = paused, .should_quit = true, .seek_delta = 0};
    }
    if (key_ascii == ' ') {
      paused = !paused;
      if (!paused) {
        return {.paused = paused, .should_quit = false, .seek_delta = 0};
      }
      continue;
    }
    if (key_ascii == 'k' || contains_key(kRightKeys, key)) {
      return {
          .paused = paused, .should_quit = false, .seek_delta = seek_frames};
    }
    if (key_ascii == 'j' || contains_key(kLeftKeys, key)) {
      return {
          .paused = paused, .should_quit = false, .seek_delta = -seek_frames};
    }

    return {.paused = paused, .should_quit = false, .seek_delta = 0};
  }
}

int seek_video(cv::VideoCapture &capture, int current_frame_index,
               int frame_delta, int total_frames) {
  int target_frame = current_frame_index + frame_delta;
  if (total_frames > 0) {
    target_frame = std::min(target_frame, total_frames - 1);
  }
  target_frame = std::max(0, target_frame);
  capture.set(cv::CAP_PROP_POS_FRAMES, target_frame);
  return target_frame;
}

std::vector<TrackRecord> detect_current_records(
    onnx::yolo::Runner &runner, const cv::Mat &frame,
    video_processing::ImageHandler &image_handler, const TrackerConfig &config,
    const FrameMetadata &metadata, int frame_index, double timestamp) {
  video_processing::LetterBoxResult letterbox;
  cv::Mat model_ready = make_model_ready_frame(
      frame, image_handler, config.model_image_size, letterbox);
  Ort::Value output = runner.run(model_ready);
  std::vector<std::vector<onnx::yolo::Detection>> decoded =
      runner.decode(output, config.conf_threshold);
  std::vector<std::vector<onnx::yolo::Detection>> nms_detections =
      runner.nms(decoded, config.iou_threshold);

  std::vector<TrackRecord> records;
  if (nms_detections.empty()) {
    return records;
  }

  records.reserve(nms_detections.front().size());
  std::ranges::transform(nms_detections.front(), std::back_inserter(records),
                         [&](const onnx::yolo::Detection &detection) {
                           return detection_to_track_record(
                               detection, letterbox, metadata, frame_index,
                               timestamp);
                         });
  return records;
}

int run_visualisation(const TrackerConfig &config) {
  cv::VideoCapture capture{config.video_path.string()};
  if (!capture.isOpened()) {
    throw std::runtime_error("Failed to open video: " +
                             config.video_path.string());
  }

  const FrameMetadata metadata = read_metadata(capture, config.video_path);
  cv::VideoWriter writer = create_writer(config, metadata);
  onnx::yolo::Runner runner{config.model_path.string().c_str()};
  video_processing::ImageHandler image_handler;
  std::deque<cv::Mat> overlay_buffer;

  const int delay_ms =
      std::max(1, static_cast<int>(std::round(1000.0 / metadata.fps)));
  bool paused = false;
  int frame_index = 0;
  int record_count = 0;

  cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
  while (true) {
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
      break;
    }

    std::vector<TrackRecord> current_records;
    if (frame_index % config.frame_stride == 0) {
      const double timestamp =
          frame_timestamp(capture, frame_index, metadata.fps);
      current_records =
          detect_current_records(runner, frame, image_handler, config, metadata,
                                 frame_index, timestamp);
      record_count += static_cast<int>(current_records.size());
    }

    cv::Mat annotated = annotate_frame(frame, current_records, metadata, config,
                                       overlay_buffer);
    cv::imshow(kWindowName, annotated);
    if (writer.isOpened()) {
      writer.write(annotated);
    }

    const PlayerAction action = wait_for_player(delay_ms, paused, metadata.fps);
    paused = action.paused;
    if (action.should_quit) {
      break;
    }

    if (action.seek_delta != 0) {
      frame_index = seek_video(capture, frame_index, action.seek_delta,
                               metadata.total_frames);
      overlay_buffer.clear();
    } else {
      ++frame_index;
    }
  }

  cv::destroyWindow(kWindowName);
  for (int i = 0; i < 5; ++i) {
    cv::waitKey(1);
  }

  return record_count;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::filesystem::path config_path = resolve_config_path(argc, argv);
    const TrackerConfig config = load_config(config_path);
    const int record_count = run_visualisation(config);
    std::cout << "TRACK_VISUALISATION_DONE records=" << record_count
              << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "movement_track failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
