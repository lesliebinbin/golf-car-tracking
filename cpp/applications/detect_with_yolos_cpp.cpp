#include "video_processing.hpp"
#include "yolos/yolos.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

namespace {

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

std::string project_file(const std::filesystem::path &relative_path) {
  return (project_root() / relative_path).lexically_normal().string();
}

} // namespace

int main(void) {
  const auto frame = cv::imread(project_file("python/video_frames/frame_0140.jpg"));
  if (frame.empty()) {
    throw std::runtime_error("Failed to read sample frame");
  }

  auto detector = yolos::YOLODetector(project_file("python/golf-car-static.onnx"),
                                      project_file("python/labels.txt"));

  auto detections = detector.detect(frame);
  std::cout << "Detections: " << detections.size() << std::endl;

  return 0;
}
