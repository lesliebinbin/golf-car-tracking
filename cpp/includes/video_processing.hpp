#pragma once

#include <opencv2/opencv.hpp>

namespace video_processing {
  int play_video(const char *video_path);
  std::vector<cv::Mat> extract_frames(const char *video_path, int frame_interval, int max_frames, int start_offset);
}
