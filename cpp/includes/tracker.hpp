#pragma once
#include "onnx.hpp"
#include <ByteTrack/BYTETracker.h>
#include <vector>
namespace tracker {
class ObjectMovement {
private:
  const int img_height;
  const int img_width;
  byte_track::BYTETracker &tracker;

public:
  byte_track::Object
  detection_to_object(const onnx::yolo::Detection &detection);
  ObjectMovement(byte_track::BYTETracker &tracker,
                 const int img_height = 640, const int img_width = 640);
  std::vector<byte_track::BYTETracker::STrackPtr>
  update(const std::vector<byte_track::Object> &objects);
  std::vector<byte_track::Object>
  detection_to_objects(const std::vector<onnx::yolo::Detection> &detections,
                       int origin_height, int origin_width, int letterbox_pad_x,
                       int letterbox_pad_y, float letterbox_scale);
};
} // namespace tracker
