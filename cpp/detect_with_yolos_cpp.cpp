#include "video_processing.hpp"
#include "yolos/yolos.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>


int main(void) {
  const auto frame = cv::imread("../python/video_frames/frame_0138.jpg");
  auto detector = yolos::YOLODetector("../python/golf-car-static.onnx",
                                      "../python/labels.txt");

  auto detections = detector.detect(frame);
  std::cout << "Detections: " << detections.size() << std::endl;

  return 0;
}
