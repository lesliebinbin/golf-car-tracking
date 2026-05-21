from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import golfcar_backend.onnx_yolo as onnx_yolo
    import golfcar_backend.video_processing as video_processing
except ImportError as exc:
    raise RuntimeError(
        "Failed to import golfcar_backend native modules. Build/install the "
        "C++ pybind modules into python/golfcar_backend first."
    ) from exc


DEFAULT_MODEL_IMAGE_SIZE = 640
WINDOW_NAME = "Golf Car Movement Tracking"
KEY_LEFT_ARROW = {2, 81, 63234, 65361, 2424832}
KEY_RIGHT_ARROW = {3, 83, 63235, 65363, 2555904}
SEEK_SECONDS = 1.0


@dataclass(frozen=True)
class TrackRecord:
    frame_index: int
    timestamp: float
    center_x: float
    center_y: float
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameMetadata:
    width: int
    height: int
    fps: float
    total_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO ONNX golf-car detection on a video and overlay a "
            "bottom-right movement-track canvas."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--canvas-size", type=int, default=180)
    parser.add_argument("--canvas-margin", type=int, default=16)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=20,
        help="Number of recent tracker canvases to average into the overlay.",
    )
    parser.add_argument(
        "--decay-strategy",
        choices=["simple-average"],
        default="simple-average",
        help="How buffered tracker canvases are blended.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Deprecated; tracker persistence is controlled by --buffer-size.",
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--model-image-size",
        type=int,
        default=DEFAULT_MODEL_IMAGE_SIZE,
        help="Square image size used for YOLO letterbox preprocessing.",
    )
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
    if args.canvas_size <= 0:
        raise ValueError("--canvas-size must be positive")
    if args.canvas_margin < 0:
        raise ValueError("--canvas-margin must be non-negative")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.trail_length < 0:
        raise ValueError("--trail-length must be non-negative")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.model_image_size <= 0:
        raise ValueError("--model-image-size must be positive")


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


def create_writer(
    output_path: Path | None, metadata: FrameMetadata
) -> cv2.VideoWriter | None:
    if output_path is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, metadata.fps, (metadata.width, metadata.height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")
    return writer


def preprocess_frame(
    frame: np.ndarray,
    image_handler: video_processing.ImageHandler,
    model_image_size: int,
) -> tuple[np.ndarray, video_processing.LetterBoxResult]:
    letterbox = image_handler.letterbox(frame, model_image_size)
    rgb = letterbox.image[:, :, ::-1]
    model_ready = (rgb.astype(np.float32) / 255.0).copy()
    return model_ready, letterbox


def predict_detections(
    runner: onnx_yolo.Runner,
    model_ready_frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> Sequence[onnx_yolo.Detection]:
    batch_detections = runner.predict(model_ready_frame, conf_threshold, iou_threshold)
    if not batch_detections:
        return []

    return batch_detections[0]


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def detection_to_track_record(
    detection: onnx_yolo.Detection,
    letterbox: video_processing.LetterBoxResult,
    metadata: FrameMetadata,
    frame_index: int,
    timestamp: float,
) -> TrackRecord:
    center_x = (detection.x - letterbox.pad_x) / letterbox.scale
    center_y = (detection.y - letterbox.pad_y) / letterbox.scale
    width = detection.w / letterbox.scale
    height = detection.h / letterbox.scale

    center_x = clamp(center_x, 0.0, float(metadata.width - 1))
    center_y = clamp(center_y, 0.0, float(metadata.height - 1))
    width = clamp(width, 0.0, float(metadata.width))
    height = clamp(height, 0.0, float(metadata.height))

    return TrackRecord(
        frame_index=frame_index,
        timestamp=timestamp,
        center_x=center_x,
        center_y=center_y,
        confidence=float(detection.confidence),
        bbox=(center_x, center_y, width, height),
    )


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


def create_tracker_canvas(
    current_records: Sequence[TrackRecord],
    metadata: FrameMetadata,
    size: int,
) -> np.ndarray:
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

    current_points = [
        map_to_canvas(record, metadata, size) for record in current_records
    ]
    for point in current_points:
        cv2.circle(canvas, point, 5, (0, 0, 255), -1)
        cv2.circle(canvas, point, 7, (255, 255, 255), 1)

    return canvas


def blend_tracker_canvases(
    overlay_buffer: Sequence[np.ndarray],
    decay_strategy: str,
) -> np.ndarray:
    if not overlay_buffer:
        raise ValueError("overlay_buffer must not be empty")
    if decay_strategy != "simple-average":
        raise ValueError(f"Unsupported decay strategy: {decay_strategy}")

    stacked = np.stack(overlay_buffer).astype(np.float32)
    return np.mean(stacked, axis=0).astype(np.uint8)


def draw_tracking_canvas(
    frame: np.ndarray,
    current_records: Sequence[TrackRecord],
    metadata: FrameMetadata,
    canvas_size: int,
    canvas_margin: int,
    overlay_buffer: deque[np.ndarray],
    decay_strategy: str,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    size = min(
        canvas_size, frame_width - 2 * canvas_margin, frame_height - 2 * canvas_margin
    )
    if size <= 20:
        return

    canvas = create_tracker_canvas(current_records, metadata, size)
    if overlay_buffer and overlay_buffer[0].shape != canvas.shape:
        overlay_buffer.clear()
    overlay_buffer.append(canvas)
    blended_canvas = blend_tracker_canvases(overlay_buffer, decay_strategy)

    x1 = frame_width - canvas_margin - size
    y1 = frame_height - canvas_margin - size
    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]
    blended = cv2.addWeighted(roi, 0.35, blended_canvas, 0.65, 0.0)
    frame[y1:y2, x1:x2] = blended


def annotate_frame(
    frame: np.ndarray,
    current_records: Sequence[TrackRecord],
    metadata: FrameMetadata,
    canvas_size: int,
    canvas_margin: int,
    overlay_buffer: deque[np.ndarray],
    decay_strategy: str,
) -> np.ndarray:
    annotated = frame.copy()
    for current_record in current_records:
        draw_detection(annotated, current_record)
    draw_tracking_canvas(
        annotated,
        current_records,
        metadata,
        canvas_size,
        canvas_margin,
        overlay_buffer,
        decay_strategy,
    )
    return annotated


def frame_timestamp(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if np.isfinite(timestamp_ms) and timestamp_ms > 0.0:
        return timestamp_ms / 1000.0
    return frame_index / fps


def wait_for_player(delay_ms: int, paused: bool, fps: float) -> tuple[bool, bool, int]:
    seek_frames = max(1, int(round(fps * SEEK_SECONDS)))
    while True:
        key = cv2.waitKeyEx(0 if paused else delay_ms)
        key_ascii = key & 0xFF
        if key_ascii in (27, ord("q")):
            return paused, True, 0
        if key_ascii == ord(" "):
            paused = not paused
            if not paused:
                return paused, False, 0
            continue
        if key_ascii == ord("k") or key in KEY_RIGHT_ARROW:
            return paused, False, seek_frames
        if key_ascii == ord("j") or key in KEY_LEFT_ARROW:
            return paused, False, -seek_frames
        return paused, False, 0


def seek_video(
    capture: cv2.VideoCapture,
    current_frame_index: int,
    frame_delta: int,
    total_frames: int,
) -> int:
    target_frame = current_frame_index + frame_delta
    if total_frames > 0:
        target_frame = min(target_frame, total_frames - 1)
    target_frame = max(0, target_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return target_frame


def run_visualisation(args: argparse.Namespace) -> list[TrackRecord]:
    validate_args(args)
    capture, metadata = open_video(args.video_path)
    writer = create_writer(args.output_path, metadata)
    runner = onnx_yolo.Runner(str(args.model_path))
    image_handler = video_processing.ImageHandler()
    records: list[TrackRecord] = []
    overlay_buffer: deque[np.ndarray] = deque(maxlen=args.buffer_size)

    delay_ms = max(1, int(round(1000.0 / metadata.fps)))
    paused = False
    frame_index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break

            current_records: list[TrackRecord] = []
            if frame_index % args.frame_stride == 0:
                timestamp = frame_timestamp(capture, frame_index, metadata.fps)
                model_ready, letterbox = preprocess_frame(
                    frame, image_handler, args.model_image_size
                )
                detections = predict_detections(
                    runner, model_ready, args.conf_threshold, args.iou_threshold
                )
                current_records = [
                    detection_to_track_record(
                        detection,
                        letterbox,
                        metadata,
                        frame_index,
                        timestamp,
                    )
                    for detection in detections
                ]
                records.extend(current_records)

            annotated = annotate_frame(
                frame,
                current_records,
                metadata,
                args.canvas_size,
                args.canvas_margin,
                overlay_buffer,
                args.decay_strategy,
            )

            cv2.imshow(WINDOW_NAME, annotated)
            if writer is not None:
                writer.write(annotated)

            paused, should_quit, seek_delta = wait_for_player(
                delay_ms, paused, metadata.fps
            )
            if should_quit:
                break

            if seek_delta != 0:
                frame_index = seek_video(
                    capture, frame_index, seek_delta, metadata.total_frames
                )
                overlay_buffer.clear()
            else:
                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyWindow(WINDOW_NAME)
        for _ in range(5):
            cv2.waitKey(1)

    return records


def main() -> None:
    args = parse_args()
    records = run_visualisation(args)
    print(f"TRACK_VISUALISATION_DONE records={len(records)}")


if __name__ == "__main__":
    main()
