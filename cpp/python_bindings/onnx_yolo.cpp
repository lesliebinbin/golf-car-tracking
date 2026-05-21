#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <format>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mat_numpy_converter.hpp"
#include "onnx.hpp"

namespace py = pybind11;
namespace converters = python_bindings;

std::string check_environment() {
  std::ostringstream info;
  info << std::format("<ONNX Runtime Version: {}>",
                      OrtGetApiBase()->GetVersionString());
  return info.str();
}

static std::vector<cv::Mat> numpy_float32_to_mat_batch(py::handle input) {
  auto array =
      py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
          input);
  if (!array) {
    throw std::runtime_error("input_frames must be a float32 NumPy array");
  }

  const py::buffer_info info = array.request();
  if (info.ndim == 3) {
    return {converters::numpy_float32_to_mat(input)};
  }

  if (info.ndim != 4) {
    throw std::runtime_error(
        "input_frames must have shape HxWxC or NxHxWxC");
  }

  const int batch_size = static_cast<int>(info.shape[0]);
  const int rows = static_cast<int>(info.shape[1]);
  const int cols = static_cast<int>(info.shape[2]);
  const int channels = static_cast<int>(info.shape[3]);

  if (batch_size <= 0 || rows <= 0 || cols <= 0) {
    throw std::runtime_error("input_frames must not be empty");
  }

  if (channels != 3) {
    throw std::runtime_error("input_frames must have 3 channels");
  }

  const float *data = static_cast<const float *>(info.ptr);
  const std::size_t frame_values =
      static_cast<std::size_t>(rows * cols * channels);

  std::vector<cv::Mat> frames;
  frames.reserve(static_cast<std::size_t>(batch_size));
  for (int batch = 0; batch < batch_size; ++batch) {
    cv::Mat frame(rows, cols, CV_32FC3,
                  const_cast<float *>(data + batch * frame_values));
    frames.push_back(frame.clone());
  }

  return frames;
}

static py::dict detection_to_dict(const onnx::yolo::Detection &detection) {
  py::dict d;
  d["x"] = detection.x;
  d["y"] = detection.y;
  d["w"] = detection.w;
  d["h"] = detection.h;
  d["class_id"] = detection.class_id;
  d["confidence"] = detection.confidence;
  d["box_format"] = "cxcywh";
  return d;
}

static onnx::yolo::Detection detection_from_dict(py::dict d) {
  return {
      .x = d["x"].cast<float>(),
      .y = d["y"].cast<float>(),
      .w = d["w"].cast<float>(),
      .h = d["h"].cast<float>(),
      .class_id = d["class_id"].cast<int>(),
      .confidence = d["confidence"].cast<float>(),
  };
}

PYBIND11_MODULE(onnx_yolo, m) {
  m.doc() = "ONNX YOLO Inference Acceleration";

  m.def("check_env", &check_environment, "ONNX Runtime dependency check");

  py::class_<onnx::yolo::Detection>(m, "Detection")
      .def(py::init<>())
      .def(py::init<float, float, float, float, int, float>(), py::arg("x"),
           py::arg("y"), py::arg("w"), py::arg("h"), py::arg("class_id"),
           py::arg("confidence"))
      .def_readwrite("x", &onnx::yolo::Detection::x)
      .def_readwrite("y", &onnx::yolo::Detection::y)
      .def_readwrite("w", &onnx::yolo::Detection::w)
      .def_readwrite("h", &onnx::yolo::Detection::h)
      .def_readwrite("class_id", &onnx::yolo::Detection::class_id)
      .def_readwrite("confidence", &onnx::yolo::Detection::confidence)
      .def("to_dict", &detection_to_dict)
      .def_static("from_dict", &detection_from_dict, py::arg("data"))
      .def("__repr__", [](const onnx::yolo::Detection &detection) {
        return py::str("Detection(x={:.3f}, y={:.3f}, w={:.3f}, h={:.3f}, "
                       "class_id={}, confidence={:.3f}, "
                       "box_format='cxcywh')")
            .format(detection.x, detection.y, detection.w, detection.h,
                    detection.class_id, detection.confidence);
      });

  py::class_<onnx::yolo::Runner>(m, "Runner")
      .def(py::init<const char *>(), py::arg("model_path"))
      .def(
          "predict",
          [](onnx::yolo::Runner &self, py::handle input_frames,
             float conf_threshold, float iou_threshold) {
            std::vector<cv::Mat> frames =
                numpy_float32_to_mat_batch(input_frames);
            Ort::Value output = self.run(frames);
            std::vector<std::vector<onnx::yolo::Detection>> decoded =
                self.decode(output, conf_threshold);
            return self.nms(decoded, iou_threshold);
          },
          py::arg("input_frames"), py::arg("conf_threshold") = 0.25f,
          py::arg("iou_threshold") = 0.7f,
          "Run ONNX YOLO inference on HxWx3 or NxHxWx3 float32 model-ready "
          "input and return batch detections.")
      .def(
          "nms",
          py::overload_cast<const std::vector<onnx::yolo::Detection> &, float>(
              &onnx::yolo::Runner::nms),
          py::arg("detections"), py::arg("iou_threshold") = 0.7f)
      .def(
          "nms",
          py::overload_cast<
              const std::vector<std::vector<onnx::yolo::Detection>> &, float>(
              &onnx::yolo::Runner::nms),
          py::arg("batch_detections"), py::arg("iou_threshold") = 0.7f);
}
