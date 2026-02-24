import argparse
import os
import re
import sys
from datetime import datetime

import cv2
import easyocr
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
CONF_THRESHOLD = 0.35
OCR_CONF_THRESHOLD = 0.25

# COCO class ids from YOLOv8n for road objects (approximate ROI for ANPR prototype)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# simple format cleanup for plate-like strings
PLATE_ALLOWED = re.compile(r"[^A-Z0-9-]")


def normalize_plate(text: str) -> str:
    cleaned = PLATE_ALLOWED.sub("", text.upper())
    return cleaned.strip("-")


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Try sensible OpenCV backends for cross-platform webcam opening."""
    candidates = [
        cv2.VideoCapture(camera_index, cv2.CAP_DSHOW),
        cv2.VideoCapture(camera_index),
    ]

    for cap in candidates:
        if cap.isOpened():
            return cap
        cap.release()

    raise RuntimeError(
        f"Could not open webcam index {camera_index}. "
        "Try another index with --camera 1 (or 2), or close apps using the camera."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANPR prototype with YOLOv8 + EasyOCR")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to YOLO model (.pt). Default: yolov8n.pt next to script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found at {args.model}. Put yolov8n.pt next to this script or pass --model."
        )

    print(f"Python executable: {sys.executable}")
    print(f"OpenCV version: {cv2.__version__}")

    yolo_model = YOLO(args.model)
    ocr_reader = easyocr.Reader(["en"], gpu=False)

    cap = open_camera(args.camera)

    print("Starting webcam. Press 'q' to quit.")

    # avoid spamming the same plate every frame
    last_seen = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed. Exiting.")
                break

            results = yolo_model(frame, verbose=False)
            h, w = frame.shape[:2]

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in VEHICLE_CLASS_IDS or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # clamp box to image bounds
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(1, min(x2, w))
                y2 = max(1, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # OCR directly on array (faster than writing temporary image files)
                ocr_result = ocr_reader.readtext(crop)

                texts = [normalize_plate(r[1]) for r in ocr_result if r[2] >= OCR_CONF_THRESHOLD]
                texts = [t for t in texts if len(t) >= 4]

                plate_text = texts[0] if texts else ""

                # draw bbox + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"vehicle {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                if plate_text:
                    now = datetime.now()
                    last_time = last_seen.get(plate_text)

                    # print once every 3s for same text
                    if not last_time or (now - last_time).total_seconds() >= 3:
                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Plate: {plate_text} | Time: {timestamp}")
                        last_seen[plate_text] = now

                    cv2.putText(
                        frame,
                        plate_text,
                        (x1, min(h - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("ANPR Webcam Prototype", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
