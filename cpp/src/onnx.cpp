#include <onnx.hpp>

#include <array>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char *require_model_path(const char *model_path) {
  if (model_path == nullptr || model_path[0] == '\0') {
    throw std::invalid_argument("model_path must not be empty");
  }
  return model_path;
}

Ort::SessionOptions create_session_options() {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(1);
  return options;
}

std::optional<std::string> lookup_metadata(const Ort::Session &session,
                                           const char *key) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata metadata = session.GetModelMetadata();
  auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
  if (!value) {
    return std::nullopt;
  }
  return std::string(value.get());
}

std::vector<int> extract_integers(const std::string &value) {
  std::vector<int> numbers;
  std::string current;

  for (char ch : value) {
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      current.push_back(ch);
      continue;
    }

    if (!current.empty()) {
      numbers.push_back(std::stoi(current));
      current.clear();
    }
  }

  if (!current.empty()) {
    numbers.push_back(std::stoi(current));
  }

  return numbers;
}

std::pair<int, int> parse_imgsz_metadata(const Ort::Session &session) {
  std::optional<std::string> imgsz = lookup_metadata(session, "imgsz");
  if (!imgsz) {
    return {640, 640};
  }

  std::vector<int> numbers = extract_integers(*imgsz);
  if (numbers.size() >= 2 && numbers[0] > 0 && numbers[1] > 0) {
    return {numbers[0], numbers[1]};
  }

  return {640, 640};
}

bool parse_dynamic_batch_metadata(const Ort::Session &session,
                                  bool shape_is_dynamic) {
  std::optional<std::string> args = lookup_metadata(session, "args");
  if (!args) {
    return shape_is_dynamic;
  }

  const bool says_dynamic_true =
      args->find("'dynamic': True") != std::string::npos ||
      args->find("\"dynamic\": true") != std::string::npos ||
      args->find("dynamic=True") != std::string::npos;
  const bool says_dynamic_false =
      args->find("'dynamic': False") != std::string::npos ||
      args->find("\"dynamic\": false") != std::string::npos ||
      args->find("dynamic=False") != std::string::npos;

  if (says_dynamic_true) {
    return true;
  }

  if (says_dynamic_false) {
    return false;
  }

  return shape_is_dynamic;
}

int64_t parse_batch_metadata(const Ort::Session &session,
                             int64_t shape_batch) {
  std::optional<std::string> batch = lookup_metadata(session, "batch");
  if (!batch) {
    return shape_batch;
  }

  std::vector<int> numbers = extract_integers(*batch);
  if (!numbers.empty() && numbers.front() > 0) {
    return numbers.front();
  }

  return shape_batch;
}

void validate_frame(const cv::Mat &frame, int channels, int height, int width) {
  if (frame.empty()) {
    throw std::invalid_argument("input frame must not be empty");
  }

  if (frame.rows != height || frame.cols != width) {
    throw std::invalid_argument(
        "input frame size does not match YOLO ONNX input metadata");
  }

  if (frame.channels() != channels) {
    throw std::invalid_argument(
        "input frame channel count does not match YOLO ONNX input metadata");
  }

  if (frame.depth() != CV_32F) {
    throw std::invalid_argument(
        "input frame must be CV_32F and already normalized for the model");
  }
}

std::vector<float> frames_to_tensor_data(const std::vector<cv::Mat> &frames,
                                         int channels, int height, int width) {
  std::vector<float> tensor_data;
  tensor_data.reserve(frames.size() * channels * height * width);

  for (const cv::Mat &frame : frames) {
    validate_frame(frame, channels, height, width);
    cv::Mat contiguous = frame.isContinuous() ? frame : frame.clone();

    for (int channel = 0; channel < channels; ++channel) {
      for (int row = 0; row < height; ++row) {
        const cv::Vec3f *row_ptr = contiguous.ptr<cv::Vec3f>(row);
        for (int col = 0; col < width; ++col) {
          tensor_data.push_back(row_ptr[col][channel]);
        }
      }
    }
  }

  return tensor_data;
}

} // namespace

onnx::yolo::Runner::Runner(const char *model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "golfcar_yolo_onnx"),
      session_options(create_session_options()),
      session(env, require_model_path(model_path), session_options),
      input_element_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED),
      dynamic_batch(false), model_batch(-1), input_channels(3),
      input_height(640), input_width(640) {
  Ort::AllocatorWithDefaultOptions allocator;

  input_name = session.GetInputNameAllocated(0, allocator).get();
  output_name = session.GetOutputNameAllocated(0, allocator).get();

  auto input_type_info = session.GetInputTypeInfo(0);
  auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = tensor_info.GetShape();
  if (shape.size() != 4) {
    throw std::runtime_error("YOLO ONNX input must have shape [N, C, H, W]");
  }

  input_element_type = tensor_info.GetElementType();
  if (input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("Only float32 YOLO ONNX inputs are supported");
  }

  dynamic_batch = parse_dynamic_batch_metadata(session, shape[0] <= 0);
  model_batch = dynamic_batch ? -1 : parse_batch_metadata(session, shape[0]);
  if (shape[1] > 0) {
    input_channels = static_cast<int>(shape[1]);
  }

  if (input_channels != 3) {
    throw std::runtime_error("YOLO ONNX input must have 3 channels");
  }

  const auto [metadata_height, metadata_width] = parse_imgsz_metadata(session);
  input_height = shape[2] > 0 ? static_cast<int>(shape[2]) : metadata_height;
  input_width = shape[3] > 0 ? static_cast<int>(shape[3]) : metadata_width;

  if (input_height <= 0 || input_width <= 0) {
    throw std::runtime_error("YOLO ONNX input height and width must be positive");
  }
}

Ort::Value onnx::yolo::Runner::run(const cv::Mat &input_frame) {
  return run(std::vector<cv::Mat>{input_frame});
}

Ort::Value onnx::yolo::Runner::run(const std::vector<cv::Mat> &input_frames) {
  if (input_frames.empty()) {
    throw std::invalid_argument("input_frames must not be empty");
  }

  const int64_t batch_size = static_cast<int64_t>(input_frames.size());
  if (!dynamic_batch && model_batch > 0 && model_batch != batch_size) {
    throw std::invalid_argument("input_frames size does not match ONNX model "
                                "static batch size");
  }

  std::vector<float> input_data = frames_to_tensor_data(
      input_frames, input_channels, input_height, input_width);
  std::array<int64_t, 4> input_shape = {batch_size, input_channels,
                                        input_height, input_width};

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(), input_shape.data(),
      input_shape.size());

  const char *input_names[] = {input_name.c_str()};
  const char *output_names[] = {output_name.c_str()};
  std::vector<Ort::Value> outputs =
      session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                  output_names, 1);

  if (outputs.empty()) {
    throw std::runtime_error("ONNX Runtime returned no outputs");
  }

  return std::move(outputs.front());
}
