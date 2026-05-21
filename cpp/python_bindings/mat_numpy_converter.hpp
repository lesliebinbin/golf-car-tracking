#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace python_bindings {
namespace py = pybind11;

cv::Mat numpy_uint8_to_mat(py::handle input);
cv::Mat numpy_float32_to_mat(py::handle input);
py::array_t<std::uint8_t> mat_to_numpy(const cv::Mat &mat);
py::array_t<std::uint8_t>
frames_to_numpy_stack(const std::vector<cv::Mat> &frames);

} // namespace python_bindings
