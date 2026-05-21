from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

CANVAS_SIZE = 180
CANVAS_MARGIN = 16
MODEL_IMAGE_SIZE = 640
TRAIL_LENGTH = 0
WINDOW_NAME = "Golf Car Movement Tracking - Ultralytics"


@dataclass(frozen=True)
class TrackRecord:
    frame_index: int
    timestamp: float
    center_x: float
    center_y: float
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int


@dataclass(frozen=True)
class FrameMetadata:
    width: int
    height: int
    fps: float
    total_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure-Python Ultralytics YOLO golf-car detection on a video "
            "and overlay a bottom-right movement-track canvas."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video path does not exist: {args.video_path}")
    if not 0.0 <= args.conf_threshold <= 1.0:
        raise ValueError("--conf-threshold must be in [0, 1]")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in [0, 1]")


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, FrameMetadata]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid dimensions: {video_path}")
    if not np.isfinite(fps) or fps <= 0.0 or fps > 240.0:
        fps = 30.0

    return capture, FrameMetadata(
        width=width, height=height, fps=fps, total_frames=max(total_frames, 0)
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def predict_track_records(
    model: YOLO,
    frame: np.ndarray,
    frame_index: int,
    timestamp: float,
    conf_threshold: float,
    iou_threshold: float,
) -> list[TrackRecord]:
    results = model.predict(
        source=frame,
        imgsz=MODEL_IMAGE_SIZE,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    records: list[TrackRecord] = []
    frame_height, frame_width = frame.shape[:2]
    for index in range(len(boxes)):
        center_x, center_y, width, height = (
            boxes.xywh[index].detach().cpu().numpy().astype(float).tolist()
        )
        confidence = float(boxes.conf[index].item())
        detected_class_id = int(boxes.cls[index].item())

        center_x = clamp(center_x, 0.0, float(frame_width - 1))
        center_y = clamp(center_y, 0.0, float(frame_height - 1))
        width = clamp(width, 0.0, float(frame_width))
        height = clamp(height, 0.0, float(frame_height))

        records.append(
            TrackRecord(
                frame_index=frame_index,
                timestamp=timestamp,
                center_x=center_x,
                center_y=center_y,
                confidence=confidence,
                bbox=(center_x, center_y, width, height),
                class_id=detected_class_id,
            )
        )

    return records


def visible_records(
    records: Sequence[TrackRecord], trail_length: int
) -> Sequence[TrackRecord]:
    if trail_length == 0 or len(records) <= trail_length:
        return records
    return records[-trail_length:]


def draw_detection(frame: np.ndarray, record: TrackRecord) -> None:
    center_x, center_y, width, height = record.bbox
    x1 = int(round(center_x - width / 2.0))
    y1 = int(round(center_y - height / 2.0))
    x2 = int(round(center_x + width / 2.0))
    y2 = int(round(center_y + height / 2.0))

    frame_height, frame_width = frame.shape[:2]
    x1 = int(clamp(x1, 0, frame_width - 1))
    y1 = int(clamp(y1, 0, frame_height - 1))
    x2 = int(clamp(x2, 0, frame_width - 1))
    y2 = int(clamp(y2, 0, frame_height - 1))
    center = (int(round(center_x)), int(round(center_y)))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, center, 4, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"Golf Car {record.confidence:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def map_to_canvas(
    record: TrackRecord, metadata: FrameMetadata, canvas_size: int
) -> tuple[int, int]:
    x = int(round(record.center_x / max(1, metadata.width - 1) * (canvas_size - 1)))
    y = int(round(record.center_y / max(1, metadata.height - 1) * (canvas_size - 1)))
    return (
        int(clamp(x, 0, canvas_size - 1)),
        int(clamp(y, 0, canvas_size - 1)),
    )


def draw_tracking_canvas(
    frame: np.ndarray,
    records: Sequence[TrackRecord],
    metadata: FrameMetadata,
    canvas_size: int,
    canvas_margin: int,
    trail_length: int,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    size = min(
        canvas_size, frame_width - 2 * canvas_margin, frame_height - 2 * canvas_margin
    )
    if size <= 20:
        return

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), (180, 180, 180), 1)
    cv2.putText(
        canvas,
        "Track",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    points = [
        map_to_canvas(record, metadata, size)
        for record in visible_records(records, trail_length)
    ]
    if len(points) >= 2:
        cv2.polylines(
            canvas, [np.array(points, dtype=np.int32)], False, (0, 180, 255), 2
        )

    for point in points[:-1]:
        cv2.circle(canvas, point, 2, (0, 120, 255), -1)
    if points:
        cv2.circle(canvas, points[-1], 5, (0, 0, 255), -1)
        cv2.circle(canvas, points[-1], 7, (255, 255, 255), 1)

    x1 = frame_width - canvas_margin - size
    y1 = frame_height - canvas_margin - size
    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]
    blended = cv2.addWeighted(roi, 0.35, canvas, 0.65, 0.0)
    frame[y1:y2, x1:x2] = blended


def annotate_frame(
    frame: np.ndarray,
    current_records: Sequence[TrackRecord],
    records: Sequence[TrackRecord],
    metadata: FrameMetadata,
    canvas_size: int,
    canvas_margin: int,
    trail_length: int,
) -> np.ndarray:
    annotated = frame.copy()
    for current_record in current_records:
        draw_detection(annotated, current_record)
    draw_tracking_canvas(
        annotated,
        records,
        metadata,
        canvas_size,
        canvas_margin,
        trail_length,
    )
    return annotated


def frame_timestamp(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if np.isfinite(timestamp_ms) and timestamp_ms > 0.0:
        return timestamp_ms / 1000.0
    return frame_index / fps


def wait_for_player(delay_ms: int, paused: bool) -> tuple[bool, bool]:
    while True:
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (27, ord("q")):
            return paused, True
        if key == ord(" "):
            paused = not paused
            if not paused:
                return paused, False
            continue
        return paused, False


def run_visualisation(args: argparse.Namespace) -> list[TrackRecord]:
    validate_args(args)
    capture, metadata = open_video(args.video_path)
    model = YOLO(str(args.model_path))
    records: list[TrackRecord] = []

    delay_ms = max(1, int(round(1000.0 / metadata.fps)))
    paused = False
    frame_index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break

            timestamp = frame_timestamp(capture, frame_index, metadata.fps)
            current_records = predict_track_records(
                model,
                frame,
                frame_index,
                timestamp,
                args.conf_threshold,
                args.iou_threshold,
            )
            records.extend(current_records)

            annotated = annotate_frame(
                frame,
                current_records,
                records,
                metadata,
                CANVAS_SIZE,
                CANVAS_MARGIN,
                TRAIL_LENGTH,
            )

            cv2.imshow(WINDOW_NAME, annotated)

            paused, should_quit = wait_for_player(delay_ms, paused)
            if should_quit:
                break

            frame_index += 1
    finally:
        capture.release()
        cv2.destroyWindow(WINDOW_NAME)
        for _ in range(5):
            cv2.waitKey(1)

    return records


def main() -> None:
    args = parse_args()
    records = run_visualisation(args)
    print(f"PURE_PY_TRACK_VISUALISATION_DONE records={len(records)}")


if __name__ == "__main__":
    main()
