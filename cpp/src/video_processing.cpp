#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "video_processing.hpp"

#define KEY_ESC 27
#define KEY_SPACE 32

namespace {

cv::VideoCapture open_video_capture(const char *video_path) {
  cv::VideoCapture cap{video_path};
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video: " << video_path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return cap;
}

enum tick_state { TICK_CONTINUE, TICK_PROCESS, TICK_DONE };

class FrameTickTracker {
private:
  int frame_offset;
  int frame_count;
  int frame_interval;
  int max_frames;
  int processed_count;

public:
  FrameTickTracker(int frame_offset, int frame_interval, int max_frames)
      : frame_offset(std::max(0, frame_offset)), frame_count(0),
        frame_interval(std::max(1, frame_interval)),
        max_frames(std::max(0, max_frames)), processed_count(0) {}

  tick_state tick() {
    if (frame_offset > 0) {
      --frame_offset;
      return TICK_CONTINUE;
    }

    if (processed_count >= max_frames) {
      return TICK_DONE;
    }

    const int current_index = frame_count % frame_interval;
    ++frame_count;

    if (current_index == 0) {
      ++processed_count;
      return TICK_PROCESS;
    }

    return TICK_CONTINUE;
  }
};

} // namespace

int video_processing::play_video(const char *video_path) {
  cv::VideoCapture cap = open_video_capture(video_path);

  const std::string window_name = "Video Playback";
  double fps = cap.get(cv::CAP_PROP_FPS);
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  if (!std::isfinite(fps) || fps < 1.0 || fps > 240.0) {
    fps = 30.0;
  }

  int delay_ms = std::max(1, static_cast<int>(std::round(1000.0 / fps)));
  int step_frames = std::max(1, static_cast<int>(std::round(fps * 5.0)));

  cv::namedWindow(window_name, cv::WINDOW_NORMAL);

  cv::Mat frame;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      break;
    }

    cv::imshow(window_name, frame);

    int current_key = cv::waitKeyEx(delay_ms);

    if ((current_key & 0xFF) == KEY_ESC || (current_key & 0xFF) == 'q') {
      break;
    }

    if ((current_key & 0xFF) == KEY_SPACE) {
      std::cout << "Paused. Press any key to continue." << std::endl;
      int pause_key = cv::waitKeyEx(0);
      if ((pause_key & 0xFF) == KEY_ESC || (pause_key & 0xFF) == 'q') {
        break;
      }
    } else if (current_key == 65363 || current_key == 2555904) {
      int current_frame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
      int target_frame = current_frame + step_frames;

      if (total_frames > 0) {
        target_frame = std::min(target_frame, total_frames - 1);
      }

      cap.set(cv::CAP_PROP_POS_FRAMES, target_frame);
    } else if (current_key == 65361 || current_key == 2424832) {
      int current_frame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
      int target_frame = std::max(current_frame - step_frames, 0);

      cap.set(cv::CAP_PROP_POS_FRAMES, target_frame);
    }

    if (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) < 1) {
      break;
    }
  }

  cv::destroyWindow(window_name);

  for (int i = 0; i < 5; ++i) {
    cv::waitKey(1);
  }

  return 0;
}

std::vector<cv::Mat> video_processing::extract_frames(const char *video_path,
                                                      int frame_interval,
                                                      int max_frames,
                                                      int start_offset) {
  FrameTickTracker tracker(start_offset, frame_interval, max_frames);
  cv::VideoCapture cap = open_video_capture(video_path);
  std::vector<cv::Mat> frames;

  cv::Mat frame;
  while (cap.read(frame)) {
    if (frame.empty()) {
      break;
    }

    switch (tracker.tick()) {
    case TICK_PROCESS:
      frames.push_back(frame.clone());
      break;

    case TICK_CONTINUE:
      break;

    case TICK_DONE:
      return frames;
    }
  }

  return frames;
}

video_processing::LetterBoxResult
video_processing::ImageHandler::letterbox(const cv::Mat input_image,
                                          cv::Size target_size) const {
  const auto h = input_image.rows;
  const auto w = input_image.cols;
  const auto scale = std::min(static_cast<float>(target_size.width) / w,
                              static_cast<float>(target_size.height) / h);
  const auto new_w = std::round(w * scale);
  const auto new_h = std::round(h * scale);
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, cv::Size(new_w, new_h), 0, 0,
             interpolation_flags);
  cv::Mat output_image{target_size, input_image.type(), pad_color};
  const auto pad_x =
      static_cast<int>(std::round((target_size.width - new_w) / 2.0f - 0.1f));
  const auto pad_y =
      static_cast<int>(std::round((target_size.height - new_h) / 2.0f - 0.1f));
  resized_image.copyTo(output_image(cv::Rect(pad_x, pad_y, new_w, new_h)));
  return {
      .image = output_image, .scale = scale, .pad_x = pad_x, .pad_y = pad_y};
}

cv::Mat video_processing::ImageHandler::letterbox_revert(
    video_processing::LetterBoxResult letterbox_result) {
  const auto output_image = letterbox_result.image;
  const auto scale = letterbox_result.scale;
  const auto pad_x = letterbox_result.pad_x;
  const auto pad_y = letterbox_result.pad_y;
  const auto new_w = output_image.cols - 2 * pad_x;
  const auto new_h = output_image.rows - 2 * pad_y;
  cv::Mat cropped_image = output_image(cv::Rect(pad_x, pad_y, new_w, new_h));
  cv::Mat original_image;
  cv::resize(cropped_image, original_image,
             cv::Size(std::round(new_w / scale), std::round(new_h / scale)), 0,
             0, interpolation_flags);
  return original_image;
}
