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
    throw std::runtime_error(
        "Only 1-channel or 3-channel frames are supported");
  }

  for (std::size_t i = 0; i < frames.size(); ++i) {
    const cv::Mat &f = frames[i];

    if (f.empty()) {
      throw std::runtime_error("Encountered empty frame");
    }

    if (f.rows != rows || f.cols != cols || f.channels() != channels ||
        f.depth() != CV_8U) {
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

  // grayscale
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

static cv::Mat numpy_uint8_to_mat(py::handle input) {
  auto array =
      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::
          ensure(input);
  if (!array) {
    throw std::runtime_error("input_image must be a uint8 NumPy array");
  }

  const py::buffer_info info = array.request();
  if (info.ndim != 2 && info.ndim != 3) {
    throw std::runtime_error("input_image must have shape HxW or HxWxC");
  }

  const int rows = static_cast<int>(info.shape[0]);
  const int cols = static_cast<int>(info.shape[1]);
  const int channels = info.ndim == 2 ? 1 : static_cast<int>(info.shape[2]);

  if (rows <= 0 || cols <= 0) {
    throw std::runtime_error("input_image must not be empty");
  }

  if (channels != 1 && channels != 3 && channels != 4) {
    throw std::runtime_error("input_image channels must be 1, 3, or 4");
  }

  cv::Mat mat(rows, cols, CV_MAKETYPE(CV_8U, channels), info.ptr);
  return mat.clone();
}

static py::array_t<std::uint8_t> mat_to_numpy(const cv::Mat &mat) {
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
  std::memcpy(out.mutable_data(), contiguous.data,
              static_cast<std::size_t>(contiguous.total() *
                                       contiguous.elemSize()));
  return out;
}

static cv::Size parse_target_size(py::object target_size) {
  cv::Size size;

  if (py::isinstance<py::tuple>(target_size) ||
      py::isinstance<py::list>(target_size)) {
    auto seq = py::reinterpret_borrow<py::sequence>(target_size);
    if (py::len(seq) < 2) {
      throw std::runtime_error("target_size must have at least 2 elements");
    }

    size.width = seq[0].cast<int>();
    size.height = seq[1].cast<int>();
  } else if (py::isinstance<py::int_>(target_size)) {
    const int s = target_size.cast<int>();
    size.width = s;
    size.height = s;
  } else {
    throw std::runtime_error("target_size must be tuple, list, or int");
  }

  if (size.width <= 0 || size.height <= 0) {
    throw std::runtime_error("target_size width and height must be positive");
  }

  return size;
}

namespace pybind11 {
namespace detail {
template <> struct type_caster<cv::Scalar> {
public:
  PYBIND11_TYPE_CASTER(cv::Scalar, _("cv::Scalar"));

  // Python -> C++: 从 tuple/list 转换
  bool load(handle src, bool) {
    if (!src)
      return false;

    // 处理 None
    if (src.is_none()) {
      value = cv::Scalar(0, 0, 0, 0);
      return true;
    }

    // 处理 tuple 或 list
    if (py::isinstance<py::tuple>(src) || py::isinstance<py::list>(src)) {
      auto seq = py::reinterpret_borrow<py::sequence>(src);
      size_t size = seq.size();

      // 默认值
      value = cv::Scalar(0, 0, 0, 0);

      if (size > 0)
        value[0] = seq[0].cast<double>();
      if (size > 1)
        value[1] = seq[1].cast<double>();
      if (size > 2)
        value[2] = seq[2].cast<double>();
      if (size > 3)
        value[3] = seq[3].cast<double>();

      return true;
    }

    // 处理单个数字（所有通道相同）
    if (py::isinstance<py::int_>(src) || py::isinstance<py::float_>(src)) {
      double val = src.cast<double>();
      value = cv::Scalar(val, val, val, val);
      return true;
    }

    return false;
  }

  // C++ -> Python: 转换为 tuple
  static handle cast(cv::Scalar src, return_value_policy, handle) {
    return py::make_tuple(src[0], src[1], src[2], src[3]).release();
  }
};
} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(golf_backend, m) {
  m.doc() = "GolfCar Tracker C++ Backend Acceleration";

  m.def("check_env", &check_environment, "C++ Dependency Check Function");

  m.def("play_video", &video_processing::play_video, "Play video with OpenCV",
        py::arg("video_path"), py::call_guard<py::gil_scoped_release>());

  m.def("extract_frames", &extract_frames_py,
        "Extract frames from video as a stacked NumPy array",
        py::arg("video_path"), py::arg("frame_interval") = 30,
        py::arg("max_frames") = 1000, py::arg("start_offset") = 0);

  py::class_<video_processing::LetterBoxResult>(m, "LetterBoxResult")
      .def(py::init<>())
      .def_property_readonly(
          "image",
          [](const video_processing::LetterBoxResult &r) {
            return mat_to_numpy(r.image);
          },
          "The letterboxed image (numpy array)")
      .def_readwrite("scale", &video_processing::LetterBoxResult::scale,
                     "Scale factor applied to original image")
      .def_readwrite("pad_x", &video_processing::LetterBoxResult::pad_x,
                     "Horizontal padding (width)")
      .def_readwrite("pad_y", &video_processing::LetterBoxResult::pad_y,
                     "Vertical padding (height)")
      // 添加 Pythonic 的字符串表示
      .def("__repr__",
           [](const video_processing::LetterBoxResult &r) {
             return py::str("<LetterBoxResult scale={:.3f}, pad=({}, {}), "
                            "image_shape={}>")
                 .format(r.scale, r.pad_x, r.pad_y,
                         r.image.empty()
                             ? "empty"
                             : py::str("({}, {})")
                                   .format(r.image.cols, r.image.rows));
           })
      // 添加字典式访问（可选）
      .def(
          "to_dict",
          [](const video_processing::LetterBoxResult &r) {
            py::dict d;
            d["image"] = mat_to_numpy(r.image);
            d["scale"] = r.scale;
            d["pad_x"] = r.pad_x;
            d["pad_y"] = r.pad_y;
            return d;
          },
          "Convert to dictionary for easy access");

  py::class_<video_processing::ImageHandler>(m, "ImageHandler")
      // 构造函数
      .def(py::init<>())
      .def(py::init<cv::InterpolationFlags, cv::Scalar>(),
           py::arg("interpolation_flags"), py::arg("pad_color"))

      // letterbox - 使用 lambda 包装以接受 tuple/list/int
      .def(
          "letterbox",
          [](const video_processing::ImageHandler &self,
             py::array input_image,
             py::object target_size) -> video_processing::LetterBoxResult {
            cv::Mat input_mat = numpy_uint8_to_mat(input_image);
            cv::Size size = parse_target_size(target_size);
            return self.letterbox(input_mat, size);
          },
          py::arg("input_image"), py::arg("target_size"))

      // letterbox_revert
      .def("letterbox_revert",
           [](video_processing::ImageHandler &self,
              video_processing::LetterBoxResult letterbox_result) {
             return mat_to_numpy(self.letterbox_revert(letterbox_result));
           },
           py::arg("letterbox_result"))

      // properties
      .def_property("interpolation_flags",
                    &video_processing::ImageHandler::get_interpolation_flags,
                    &video_processing::ImageHandler::set_interpolation_flags)
      .def_property("pad_color", &video_processing::ImageHandler::get_pad_color,
                    &video_processing::ImageHandler::set_pad_color)

      .def("__repr__", [](const video_processing::ImageHandler &h) {
        return py::str(
                   "<ImageHandler interpolation={}, pad_color=({}, {}, {})>")
            .format(h.get_interpolation_flags(), h.get_pad_color()[0],
                    h.get_pad_color()[1], h.get_pad_color()[2]);
      });

  py::enum_<cv::InterpolationFlags>(m, "InterpolationFlags")
      .value("NEAREST", cv::INTER_NEAREST)
      .value("LINEAR", cv::INTER_LINEAR)
      .value("CUBIC", cv::INTER_CUBIC)
      .value("AREA", cv::INTER_AREA)
      .value("LANCZOS4", cv::INTER_LANCZOS4)
      .value("LINEAR_EXACT", cv::INTER_LINEAR_EXACT)
      .value("NEAREST_EXACT", cv::INTER_NEAREST_EXACT)
      .export_values()
      .def("__repr__", [](cv::InterpolationFlags flag) {
        switch (flag) {
        case cv::INTER_NEAREST:
          return "NEAREST";
        case cv::INTER_LINEAR:
          return "LINEAR";
        case cv::INTER_CUBIC:
          return "CUBIC";
        case cv::INTER_AREA:
          return "AREA";
        case cv::INTER_LANCZOS4:
          return "LANCZOS4";
        default:
          return "UNKNOWN";
        }
      });

}
