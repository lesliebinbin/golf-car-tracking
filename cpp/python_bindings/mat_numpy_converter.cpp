#include "mat_numpy_converter.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace python_bindings {

namespace {

template <typename T>
cv::Mat numpy_to_mat(py::handle input, int cv_depth, const char *error_prefix) {
  auto array =
      py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(input);
  if (!array) {
    throw std::runtime_error(std::string{error_prefix} +
                             " must be a NumPy array");
  }

  const py::buffer_info info = array.request();
  if (info.ndim != 2 && info.ndim != 3) {
    throw std::runtime_error(std::string{error_prefix} +
                             " must have shape HxW or HxWxC");
  }

  const int rows = static_cast<int>(info.shape[0]);
  const int cols = static_cast<int>(info.shape[1]);
  const int channels = info.ndim == 2 ? 1 : static_cast<int>(info.shape[2]);

  if (rows <= 0 || cols <= 0) {
    throw std::runtime_error(std::string{error_prefix} + " must not be empty");
  }

  if (channels != 1 && channels != 3 && channels != 4) {
    throw std::runtime_error(std::string{error_prefix} +
                             " channels must be 1, 3, or 4");
  }

  cv::Mat mat(rows, cols, CV_MAKETYPE(cv_depth, channels), info.ptr);
  return mat.clone();
}

} // namespace

cv::Mat numpy_uint8_to_mat(py::handle input) {
  return numpy_to_mat<std::uint8_t>(input, CV_8U, "input_image");
}

cv::Mat numpy_float32_to_mat(py::handle input) {
  return numpy_to_mat<float>(input, CV_32F, "input_image");
}

py::array_t<std::uint8_t> mat_to_numpy(const cv::Mat &mat) {
  if (mat.empty()) {
    return py::array_t<std::uint8_t>(std::vector<py::ssize_t>{0, 0, 0});
  }

  if (mat.depth() != CV_8U) {
    throw std::runtime_error("Only CV_8U images can be converted to NumPy");
  }

  const int channels = mat.channels();
  cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();

  std::vector<py::ssize_t> shape;
  if (channels == 1) {
    shape = {static_cast<py::ssize_t>(mat.rows),
             static_cast<py::ssize_t>(mat.cols)};
  } else {
    shape = {static_cast<py::ssize_t>(mat.rows),
             static_cast<py::ssize_t>(mat.cols),
             static_cast<py::ssize_t>(channels)};
  }

  py::array_t<std::uint8_t> out(shape);
  std::memcpy(
      out.mutable_data(), contiguous.data,
      static_cast<std::size_t>(contiguous.total() * contiguous.elemSize()));
  return out;
}

py::array_t<std::uint8_t>
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

  for (const cv::Mat &frame : frames) {
    if (frame.empty()) {
      throw std::runtime_error("Encountered empty frame");
    }

    if (frame.rows != rows || frame.cols != cols ||
        frame.channels() != channels || frame.depth() != CV_8U) {
      throw std::runtime_error("All frames must have same shape/type");
    }
  }

  if (channels == 3) {
    py::array_t<std::uint8_t> out({static_cast<py::ssize_t>(frames.size()),
                                   static_cast<py::ssize_t>(rows),
                                   static_cast<py::ssize_t>(cols),
                                   static_cast<py::ssize_t>(3)});

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

  py::array_t<std::uint8_t> out({static_cast<py::ssize_t>(frames.size()),
                                 static_cast<py::ssize_t>(rows),
                                 static_cast<py::ssize_t>(cols)});

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

} // namespace python_bindings
