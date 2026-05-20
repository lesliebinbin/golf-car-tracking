#include <format>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <sstream>
#include <string>
#include <video_processing.hpp>

namespace py = pybind11;

std::string check_environment() {
  std::ostringstream info;
  info << "<GolfCar Tracker C++ Backend Acceleration>";
  info << "\n" << "<ONNX Runtime APIReady>";
  info << std::format("\n<OpenCV Version: {}>", CV_VERSION);
  return info.str();
}

PYBIND11_MODULE(golf_backend, m) {
  m.doc() = "GolfCar Tracker C++ Backend Acceleration";

  m.def("check_env", &check_environment, "C++ Dependency Check Function");
  m.def("play_video",
        &video_processing::play_video,
        "Play video with OpenCV",
        py::arg("video_path"),
        py::call_guard<py::gil_scoped_release>()
        );
}
