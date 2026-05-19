#include <iostream>
#include <opencv2/opencv.hpp>
#include <video_processing.hpp>

// 统一蛇形风格的宏定义
#define KEY_ESC 27
#define KEY_SPACE 32
#define KEY_LEFT 2
#define KEY_RIGHT 3

int video_processing::play_video(const char *video_path) {
  cv::VideoCapture cap{video_path};
  if (!cap.isOpened())
    return -1;

  // 变量全部洗成标准的 snake_case
  double fps = cap.get(cv::CAP_PROP_FPS);
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int delay_ms = (fps > 0) ? static_cast<int>(1000.0 / fps) : 30;
  int step_frames = static_cast<int>(fps * 5);

  cv::Mat frame;
  while (cap.isOpened()) {
    cap >> frame;
    if (frame.empty())
      break;

    cv::imshow("Video Playback", frame);

    int current_key = cv::waitKey(delay_ms) & 0xFF;

    if (current_key == KEY_ESC || current_key == 'q') {
      break;
    }

    if (current_key == KEY_SPACE) {
      std::cout << "⏸️ 暂停中... 按任意键继续。" << std::endl;
      cv::waitKey(0);
    } else if (current_key == KEY_RIGHT) {
      int current_frame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
      int target_frame =
          std::min(current_frame + step_frames, total_frames - 1);

      cap.set(cv::CAP_PROP_POS_FRAMES, target_frame);
    } else if (current_key == KEY_LEFT) {
      int current_frame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
      int target_frame = std::max(current_frame - (step_frames + 1), 0);

      cap.set(cv::CAP_PROP_POS_FRAMES, target_frame);
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
