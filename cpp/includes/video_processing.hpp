#pragma once

#include <opencv2/opencv.hpp>

namespace video_processing {
int play_video(const char *video_path);
std::vector<cv::Mat> extract_frames(const char *video_path, int frame_interval,
                                    int max_frames, int start_offset);

struct LetterBoxResult {
  cv::Mat image;
  float scale;
  int pad_x;
  int pad_y;
};
class ImageHandler {
private:
  cv::InterpolationFlags interpolation_flags;
  cv::Scalar pad_color;

public:
  ImageHandler(cv::InterpolationFlags flags, cv::Scalar pad_color)
      : interpolation_flags(flags), pad_color(pad_color) {}

  ImageHandler() : ImageHandler(cv::INTER_LINEAR, cv::Scalar(114, 114, 114)) {}
  LetterBoxResult letterbox(const cv::Mat input_image,
                            cv::Size target_size) const;
  cv::Mat letterbox_revert(LetterBoxResult letterbox_result);
  cv::InterpolationFlags get_interpolation_flags() const {
    return interpolation_flags;
  }
  cv::Scalar get_pad_color() const { return pad_color; }
  void set_interpolation_flags(cv::InterpolationFlags interpolation_flags) {
    this->interpolation_flags = interpolation_flags;
  }
  void set_pad_color(cv::Scalar pad_color) { this->pad_color = pad_color; }
};
} // namespace video_processing
