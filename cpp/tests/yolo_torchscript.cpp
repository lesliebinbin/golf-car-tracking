#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <vector>

namespace {

constexpr int kModelImageSize = 640;
constexpr float kConfThreshold = 0.25f;

std::filesystem::path project_root() {
#ifdef GOLFCAR_REPO_ROOT
  return std::filesystem::path{GOLFCAR_REPO_ROOT}.lexically_normal();
#else
  std::filesystem::path source_path{__FILE__};
  if (source_path.is_relative()) {
    const std::filesystem::path executable_dir =
        std::filesystem::read_symlink("/proc/self/exe").parent_path();
    source_path = executable_dir / source_path;
  }

  return std::filesystem::weakly_canonical(source_path)
      .parent_path()
      .parent_path()
      .parent_path();
#endif
}

std::filesystem::path resolve_project_path(const std::string &path) {
  std::filesystem::path candidate{path};
  if (candidate.is_absolute()) {
    return candidate;
  }
  return (project_root() / candidate).lexically_normal();
}

cv::Mat letterbox(const cv::Mat &image, int target_size) {
  if (image.empty()) {
    throw std::invalid_argument("input image must not be empty");
  }

  const int width = image.cols;
  const int height = image.rows;
  const double scale =
      std::min(static_cast<double>(target_size) / static_cast<double>(width),
               static_cast<double>(target_size) / static_cast<double>(height));
  const int resized_width = static_cast<int>(std::round(width * scale));
  const int resized_height = static_cast<int>(std::round(height * scale));

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
             cv::INTER_LINEAR);

  cv::Mat canvas(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
  const int left = (target_size - resized_width) / 2;
  const int top = (target_size - resized_height) / 2;
  resized.copyTo(canvas(cv::Rect(left, top, resized_width, resized_height)));
  return canvas;
}

torch::Tensor image_to_tensor(const std::filesystem::path &image_path) {
  cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }

  cv::Mat padded = letterbox(image, kModelImageSize);
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

  torch::Tensor tensor =
      torch::from_blob(normalized.data, {kModelImageSize, kModelImageSize, 3},
                       torch::TensorOptions().dtype(torch::kFloat32));
  return tensor.permute({2, 0, 1}).unsqueeze(0).contiguous().clone();
}

std::vector<int64_t> tensor_shape(const torch::Tensor &tensor) {
  return std::vector<int64_t>{tensor.sizes().begin(), tensor.sizes().end()};
}

void print_shape(const std::vector<int64_t> &shape) {
  std::cout << "(";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << shape[index];
  }
  std::cout << ")";
}

torch::Tensor first_tensor_from_output(const c10::IValue &output) {
  if (output.isTensor()) {
    return output.toTensor();
  }

  if (output.isTuple()) {
    for (const c10::IValue &item : output.toTuple()->elements()) {
      if (item.isTensor()) {
        return item.toTensor();
      }
    }
  }

  if (output.isList()) {
    const c10::List<c10::IValue> list = output.toList();
    for (const c10::IValue item : list) {
      if (item.isTensor()) {
        return item.toTensor();
      }
    }
  }

  throw std::runtime_error("TorchScript output did not contain a tensor");
}

void assert_yolo_detect_output(const torch::Tensor &output) {
  if (!output.defined()) {
    throw std::runtime_error("model output tensor is undefined");
  }

  const torch::Tensor cpu_output = output.detach().to(torch::kCPU);
  const std::vector<int64_t> shape = tensor_shape(cpu_output);
  if (shape.size() != 3) {
    throw std::runtime_error("YOLO output must be rank 3");
  }

  const bool channels_first = shape[1] == 5;
  const bool channels_last = shape[2] == 5;
  if (!channels_first && !channels_last) {
    throw std::runtime_error("YOLO output must contain 5 detection values");
  }

  const torch::Tensor scores =
      channels_first ? cpu_output.index({torch::indexing::Slice(), 4,
                                         torch::indexing::Slice()})
                     : cpu_output.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(), 4});
  const float max_score = scores.max().item<float>();
  const int64_t confident_count =
      scores.ge(kConfThreshold).sum().item<int64_t>();

  if (!std::isfinite(max_score)) {
    throw std::runtime_error("YOLO output contains a non-finite score");
  }

  std::cout << "Output shape: ";
  print_shape(shape);
  std::cout << "\nMax score: " << max_score
            << "\nScores >= " << kConfThreshold << ": " << confident_count
            << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::filesystem::path model_path = resolve_project_path(
        argc > 1 ? argv[1] : "python/golf-car.torchscript");
    const std::filesystem::path image_path =
        resolve_project_path(argc > 2 ? argv[2]
                                      : "python/video_frames/frame_0139.jpg");

    if (!std::filesystem::is_regular_file(model_path)) {
      throw std::runtime_error("Model file does not exist: " +
                               model_path.string());
    }
    if (!std::filesystem::is_regular_file(image_path)) {
      throw std::runtime_error("Image file does not exist: " +
                               image_path.string());
    }

    torch::NoGradGuard no_grad;
    torch::jit::script::Module module = torch::jit::load(model_path.string());
    module.eval();

    torch::Tensor input = image_to_tensor(image_path);
    std::cout << "Model: " << model_path << "\nImage: " << image_path
              << "\nInput shape: ";
    print_shape(tensor_shape(input));
    std::cout << std::endl;

    const c10::IValue output = module.forward({input});
    assert_yolo_detect_output(first_tensor_from_output(output));

    std::cout << "TorchScript YOLO smoke test passed." << std::endl;
    return 0;
  } catch (const c10::Error &error) {
    std::cerr << "LibTorch error: " << error.what() << std::endl;
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
}
