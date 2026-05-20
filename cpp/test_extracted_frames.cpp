#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "video_processing.hpp"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <video_path> [frame_interval] [max_frames] [start_offset]\n";
    return EXIT_FAILURE;
  }

  const char *video_path = argv[1];
  const int frame_interval = (argc >= 3) ? std::atoi(argv[2]) : 30;
  const int max_frames = (argc >= 4) ? std::atoi(argv[3]) : 1000;
  const int start_offset = (argc >= 5) ? std::atoi(argv[4]) : 0;

  std::vector<cv::Mat> frames = video_processing::extract_frames(
      video_path, frame_interval, max_frames, start_offset);

  std::cout << "Extracted frame count: " << frames.size() << std::endl;

  if (frames.empty()) {
    std::cerr << "No frames extracted." << std::endl;
    return EXIT_FAILURE;
  }

  const std::string window_name = "Extracted Frames Test";
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);

  std::size_t idx = 0;
  while (true) {
    const cv::Mat &frame = frames[idx];
    if (frame.empty()) {
      std::cerr << "Frame " << idx << " is empty." << std::endl;
    } else {
      cv::imshow(window_name, frame);

      const cv::Vec3b &p = frame.at<cv::Vec3b>(0, 0);
      std::cout << "Showing frame[" << idx << "] "
                << "shape=(" << frame.rows << ", " << frame.cols << ", "
                << frame.channels() << ") "
                << "pixel(0,0)=(" << static_cast<int>(p[0]) << ", "
                << static_cast<int>(p[1]) << ", " << static_cast<int>(p[2])
                << ")" << std::endl;
    }

    int key = cv::waitKeyEx(0);

    // ESC or q: quit
    if ((key & 0xFF) == 27 || (key & 0xFF) == 'q') {
      break;
    }

    // Right arrow / d / space: next frame
    if (key == 65363 || key == 2555904 || (key & 0xFF) == 'd' ||
        (key & 0xFF) == ' ') {
      if (idx + 1 < frames.size()) {
        ++idx;
      }
      continue;
    }

    // Left arrow / a: previous frame
    if (key == 65361 || key == 2424832 || (key & 0xFF) == 'a') {
      if (idx > 0) {
        --idx;
      }
      continue;
    }
  }

  cv::destroyWindow(window_name);
  for (int i = 0; i < 5; ++i) {
    cv::waitKey(1);
  }

  return EXIT_SUCCESS;
}
