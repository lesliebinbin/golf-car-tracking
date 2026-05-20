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
  int current_index;

public:
  FrameTickTracker(int frame_offset, int frame_interval, int max_frames)
      : frame_offset(std::max(0, frame_offset)), frame_count(0),
        frame_interval(std::max(1, frame_interval)),
        max_frames(std::max(0, max_frames)), current_index(0) {}

  tick_state tick() {
    if (frame_offset > 0) {
      --frame_offset;
      return TICK_CONTINUE;
    }

    if (frame_count >= max_frames) {
      return TICK_DONE;
    }

    current_index = frame_count % frame_interval;
    ++frame_count;

    if (current_index == 0) {
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
