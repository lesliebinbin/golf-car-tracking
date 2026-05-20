from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["Golf Car"]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def pick_image(source: Path) -> Path:
    if source.is_file():
        return source

    images = sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    return images[0]


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width = round(width * ratio)
    new_height = round(height * ratio)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2
    left = round(pad_x - 0.1)
    top = round(pad_y - 0.1)
    canvas[top : top + new_height, left : left + new_width] = resized

    return canvas, ratio, (left, top)


def preprocess(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, ratio, pad = letterbox(image, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None, ...], ratio, pad


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = boxes.copy()
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-9)

        order = order[1:][iou <= iou_threshold]

    return keep


def parse_yolo_output(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    ratio: float,
    pad: tuple[float, float],
    original_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.squeeze(output)

    if pred.ndim != 2:
        raise ValueError(f"Expected 2D YOLO output after squeeze, got shape {output.shape}")

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] == 6:
        boxes = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(np.int32)
    else:
        boxes = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes = xywh_to_xyxy(boxes)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio

    original_height, original_width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_height)

    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], class_ids[keep]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    annotated = image.copy()

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class-{class_id}"
        label = f"{name} {score:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def providers_from_choice(choice: str) -> list[str]:
    available = set(ort.get_available_providers())
    preferred = {
        "auto": ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    }[choice]
    providers = [provider for provider in preferred if provider in available]
    if not providers:
        raise RuntimeError(f"No requested providers available. Found: {sorted(available)}")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a YOLO ONNX model on one image.")
    parser.add_argument("--model", default="golf-car-static.onnx")
    parser.add_argument("--source", default="video_frames")
    parser.add_argument("--output", default="onnx_test_output.jpg")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "coreml"], default="auto")
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    image_path = pick_image(resolve_path(args.source))
    output_path = resolve_path(args.output)

    session = ort.InferenceSession(
        str(model_path),
        providers=providers_from_choice(args.provider),
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    tensor, ratio, pad = preprocess(image, args.imgsz)
    outputs = session.run([output_name], {input_name: tensor})

    boxes, scores, class_ids = parse_yolo_output(
        outputs[0],
        args.conf,
        args.iou,
        ratio,
        pad,
        image.shape[:2],
    )

    annotated = draw_detections(image, boxes, scores, class_ids)
    cv2.imwrite(str(output_path), annotated)

    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Providers: {session.get_providers()}")
    print(f"Detections: {len(boxes)}")
    for box, score, class_id in zip(boxes, scores, class_ids):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class-{class_id}"
        print(f"- {name}: {score:.3f}, box={box.round(1).tolist()}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
