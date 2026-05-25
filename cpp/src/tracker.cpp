#include "tracker.hpp"
#include "onnx.hpp"
#include <algorithm>
#include <iterator>
#include <optional>
#include <ranges>
#include <vector>

byte_track::Object tracker::ObjectMovement::detection_to_object(
    const onnx::yolo::Detection &detection) {
  return {
      {detection.x - detection.w / 2.0f, detection.y - detection.h / 2.0f,
       detection.w, detection.h},
      detection.class_id,
      detection.confidence,
  };
}

tracker::ObjectMovement::ObjectMovement(byte_track::BYTETracker &tracker,
                                        const int img_height,
                                        const int img_width)
    : img_height(img_height), img_width(img_width), tracker{tracker} {}

std::vector<byte_track::BYTETracker::STrackPtr>
tracker::ObjectMovement::update(
    const std::vector<byte_track::Object> &objects) {
  return this->tracker.update(objects);
}

std::vector<byte_track::Object> tracker::ObjectMovement::detection_to_objects(
    const std::vector<onnx::yolo::Detection> &detections, int origin_height,
    int origin_width, int letterbox_pad_x, int letterbox_pad_y,
    float letterbox_scale) {
  auto to_object = [&](const onnx::yolo::Detection &detection)
      -> std::optional<byte_track::Object> {
    const float center_x =
        (detection.x - static_cast<float>(letterbox_pad_x)) / letterbox_scale;
    const float center_y =
        (detection.y - static_cast<float>(letterbox_pad_y)) / letterbox_scale;
    const float width = detection.w / letterbox_scale;
    const float height = detection.h / letterbox_scale;

    const float x1 = std::clamp(center_x - width / 2.0f, 0.0f,
                                static_cast<float>(origin_width - 1));
    const float y1 = std::clamp(center_y - height / 2.0f, 0.0f,
                                static_cast<float>(origin_height - 1));
    const float x2 = std::clamp(center_x + width / 2.0f, 0.0f,
                                static_cast<float>(origin_width - 1));
    const float y2 = std::clamp(center_y + height / 2.0f, 0.0f,
                                static_cast<float>(origin_height - 1));

    const float clipped_width = x2 - x1;
    const float clipped_height = y2 - y1;

    if (clipped_width <= 0.0f || clipped_height <= 0.0f) {
      return std::nullopt;
    }

    return byte_track::Object{
        byte_track::Rect<float>{x1, y1, clipped_width, clipped_height},
        detection.class_id,
        detection.confidence,
    };
  };

  auto object_options =
      detections | std::views::transform(to_object) |
      std::views::filter([](const std::optional<byte_track::Object> &object) {
        return object.has_value();
      });

  std::vector<byte_track::Object> objects;
  objects.reserve(detections.size());

  std::ranges::transform(object_options, std::back_inserter(objects),
                         [](const std::optional<byte_track::Object> &object) {
                           return object.value();
                         });

  return objects;
}
