from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import golfcar_backend.onnx_yolo as yolo
import golfcar_backend.video_processing as video_processing


ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 640


def make_model_ready_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    handler = video_processing.ImageHandler()
    letterbox = handler.letterbox(image, IMAGE_SIZE)
    rgb = letterbox.image[:, :, ::-1]
    return (rgb.astype(np.float32) / 255.0).copy()


def assert_detection_batch(batch: list[list[yolo.Detection]], expected_size: int) -> None:
    assert len(batch) == expected_size
    for detections in batch:
        assert detections
        for detection in detections:
            assert np.isfinite(
                [
                    detection.x,
                    detection.y,
                    detection.w,
                    detection.h,
                    detection.confidence,
                ]
            ).all()
            assert detection.w > 0
            assert detection.h > 0
            assert detection.class_id >= 0
            assert detection.confidence >= 0.25


def main() -> None:
    print(yolo.check_env())

    detection = yolo.Detection(1, 2, 3, 4, 0, 0.5)
    detection_dict = detection.to_dict()
    assert detection_dict["box_format"] == "cxcywh"
    assert yolo.Detection.from_dict(detection_dict).to_dict() == detection_dict
    print(detection)

    image_paths = [
        ROOT / "video_frames/frame_0138.jpg",
        ROOT / "video_frames/frame_0139.jpg",
        ROOT / "video_frames/frame_0148.jpg",
    ]
    frames = [make_model_ready_frame(path) for path in image_paths]

    static_runner = yolo.Runner(str(ROOT / "golf-car-static.onnx"))
    static_detections = static_runner.predict(frames[0])
    assert_detection_batch(static_detections, 1)
    print(f"static detections: {[len(x) for x in static_detections]}")

    synthetic = [
        yolo.Detection(100, 100, 50, 50, 0, 0.9),
        yolo.Detection(102, 102, 50, 50, 0, 0.8),
        yolo.Detection(102, 102, 50, 50, 1, 0.7),
        yolo.Detection(300, 300, 20, 20, 0, 0.6),
    ]
    synthetic_nms = static_runner.nms(synthetic, 0.5)
    assert len(synthetic_nms) == 3

    dynamic_runner = yolo.Runner(str(ROOT / "golf-car-dynamic.onnx"))
    dynamic_detections = dynamic_runner.predict(np.stack(frames, axis=0))
    assert_detection_batch(dynamic_detections, len(frames))
    print(f"dynamic detections: {[len(x) for x in dynamic_detections]}")

    print("ONNX_YOLO_SMOKETEST_OK")


if __name__ == "__main__":
    main()
