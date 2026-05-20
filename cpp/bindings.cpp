#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "video_processing.hpp"

namespace py = pybind11;

std::string check_environment() {
  std::ostringstream info;
  info << "<GolfCar Tracker C++ Backend Acceleration>";
  info << "\n"
       << "<ONNX Runtime APIReady>";
  info << std::format("\n<OpenCV Version: {}>", CV_VERSION);
  return info.str();
}


static py::array_t<std::uint8_t>
frames_to_numpy_stack(const std::vector<cv::Mat> &frames) {
  if (frames.empty()) {
    return py::array_t<std::uint8_t>(std::vector<py::ssize_t>{0, 0, 0, 0});
  }

  const cv::Mat &first = frames.front();
  if (first.empty()) {
    throw std::runtime_error("First frame is empty");
  }

  if (first.depth() != CV_8U) {
    throw std::runtime_error("Only CV_8U frames are supported");
  }

  const int rows = first.rows;
  const int cols = first.cols;
  const int channels = first.channels();

  if (channels != 3 && channels != 1) {
    throw std::runtime_error("Only 1-channel or 3-channel frames are supported");
  }

  for (std::size_t i = 0; i < frames.size(); ++i) {
    const cv::Mat &f = frames[i];

    if (f.empty()) {
      throw std::runtime_error("Encountered empty frame");
    }

    if (f.rows != rows || f.cols != cols ||
        f.channels() != channels || f.depth() != CV_8U) {
      throw std::runtime_error("All frames must have same shape/type");
    }
  }

  if (channels == 3) {
    py::array_t<std::uint8_t> out({
        static_cast<py::ssize_t>(frames.size()),
        static_cast<py::ssize_t>(rows),
        static_cast<py::ssize_t>(cols),
        static_cast<py::ssize_t>(3)
    });

    auto buf = out.mutable_unchecked<4>();

    for (py::ssize_t n = 0; n < static_cast<py::ssize_t>(frames.size()); ++n) {
      const cv::Mat &src = frames[n];
      cv::Mat contiguous = src.isContinuous() ? src : src.clone();

      for (int r = 0; r < rows; ++r) {
        const std::uint8_t *row_ptr = contiguous.ptr<std::uint8_t>(r);
        for (int c = 0; c < cols; ++c) {
          const int base = c * 3;
          buf(n, r, c, 0) = row_ptr[base + 0];
          buf(n, r, c, 1) = row_ptr[base + 1];
          buf(n, r, c, 2) = row_ptr[base + 2];
        }
      }
    }

    return out;
  }

  // grayscale
  py::array_t<std::uint8_t> out({
      static_cast<py::ssize_t>(frames.size()),
      static_cast<py::ssize_t>(rows),
      static_cast<py::ssize_t>(cols)
  });

  auto buf = out.mutable_unchecked<3>();

  for (py::ssize_t n = 0; n < static_cast<py::ssize_t>(frames.size()); ++n) {
    const cv::Mat &src = frames[n];
    cv::Mat contiguous = src.isContinuous() ? src : src.clone();

    for (int r = 0; r < rows; ++r) {
      const std::uint8_t *row_ptr = contiguous.ptr<std::uint8_t>(r);
      for (int c = 0; c < cols; ++c) {
        buf(n, r, c) = row_ptr[c];
      }
    }
  }

  return out;
}

static py::array extract_frames_py(const char *video_path, int frame_interval,
                                   int max_frames, int start_offset) {
  std::vector<cv::Mat> frames;

  {
    py::gil_scoped_release release;
    frames = video_processing::extract_frames(video_path, frame_interval,
                                              max_frames, start_offset);
  }

  return frames_to_numpy_stack(frames);
}

PYBIND11_MODULE(golf_backend, m) {
  m.doc() = "GolfCar Tracker C++ Backend Acceleration";

  m.def("check_env", &check_environment, "C++ Dependency Check Function");

  m.def("play_video", &video_processing::play_video, "Play video with OpenCV",
        py::arg("video_path"), py::call_guard<py::gil_scoped_release>());

  m.def("extract_frames", &extract_frames_py,
        "Extract frames from video as a stacked NumPy array",
        py::arg("video_path"), py::arg("frame_interval") = 30,
        py::arg("max_frames") = 1000, py::arg("start_offset") = 0);
}
