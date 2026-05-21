#include <format>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mat_numpy_converter.hpp"
#include "video_processing.hpp"

namespace py = pybind11;
namespace converters = python_bindings;

std::string check_environment() {
  std::ostringstream info;
  info << std::format("<OpenCV Version: {}>", CV_VERSION);
  return info.str();
}

static py::array extract_frames_py(const char *video_path, int frame_interval,
                                   int max_frames, int start_offset) {
  std::vector<cv::Mat> frames;

  {
    py::gil_scoped_release release;
    frames = video_processing::extract_frames(video_path, frame_interval,
                                              max_frames, start_offset);
  }

  return converters::frames_to_numpy_stack(frames);
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

PYBIND11_MODULE(video_processing, m) {
  m.doc() = "Video Processing Acceleration";

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
            return converters::mat_to_numpy(r.image);
          },
          "The letterboxed image (numpy array)")
      .def_readwrite("scale", &video_processing::LetterBoxResult::scale,
                     "Scale factor applied to original image")
      .def_readwrite("pad_x", &video_processing::LetterBoxResult::pad_x,
                     "Horizontal padding (width)")
      .def_readwrite("pad_y", &video_processing::LetterBoxResult::pad_y,
                     "Vertical padding (height)")
      // Add Pythonic string representation.
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
      // Add dictionary-style access.
      .def(
          "to_dict",
          [](const video_processing::LetterBoxResult &r) {
            py::dict d;
            d["image"] = converters::mat_to_numpy(r.image);
            d["scale"] = r.scale;
            d["pad_x"] = r.pad_x;
            d["pad_y"] = r.pad_y;
            return d;
          },
          "Convert to dictionary for easy access");

  py::class_<video_processing::ImageHandler>(m, "ImageHandler")
      // Constructors.
      .def(py::init<>())
      .def(py::init<cv::InterpolationFlags, cv::Scalar>(),
           py::arg("interpolation_flags"), py::arg("pad_color"))

      // Use a lambda wrapper to accept tuple/list/int target sizes.
      .def(
          "letterbox",
          [](const video_processing::ImageHandler &self, py::array input_image,
             py::object target_size) -> video_processing::LetterBoxResult {
            cv::Mat input_mat = converters::numpy_uint8_to_mat(input_image);
            cv::Size size = parse_target_size(target_size);
            return self.letterbox(input_mat, size);
          },
          py::arg("input_image"), py::arg("target_size"))

      // letterbox_revert
      .def(
          "letterbox_revert",
          [](video_processing::ImageHandler &self,
             video_processing::LetterBoxResult letterbox_result) {
          return converters::mat_to_numpy(
              self.letterbox_revert(letterbox_result));
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

namespace pybind11::detail {
template <> struct type_caster<cv::Scalar> {
public:
  PYBIND11_TYPE_CASTER(cv::Scalar, _("cv::Scalar"));

  bool load(handle src, bool) {
    if (!src)
      return false;

    if (src.is_none()) {
      value = cv::Scalar(0, 0, 0, 0);
      return true;
    }

    // Handle tuple or list input.
    if (py::isinstance<py::tuple>(src) || py::isinstance<py::list>(src)) {
      auto seq = py::reinterpret_borrow<py::sequence>(src);
      size_t size = seq.size();

      // Default value.
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

    if (py::isinstance<py::int_>(src) || py::isinstance<py::float_>(src)) {
      double val = src.cast<double>();
      value = cv::Scalar(val, val, val, val);
      return true;
    }

    return false;
  }

  static handle cast(cv::Scalar src, return_value_policy, handle) {
    return py::make_tuple(src[0], src[1], src[2], src[3]).release();
  }
};
} // namespace pybind11::detail
