#!/usr/bin/env python3
"""Webcam pose-estimation stream.

Captures the default webcam (or any OpenCV-compatible camera index / URL),
runs pose estimation on every frame, and shows the result in a window with
skeleton overlay.

Usage
-----
    python webcam_stream.py --model path/to/model.onnx
    python webcam_stream.py --model path/to/model.onnx --camera 1
    python webcam_stream.py --model path/to/model.onnx --conf 0.35 --iou 0.5

Controls
--------
    q / Esc  – quit
    s        – save current frame to disk
    p        – pause / resume
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from pose_estimation import PoseEstimationModel, draw_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time pose estimation from a webcam stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Path to the ONNX (.onnx) or CoreML (.mlpackage / .mlmodel) model.",
    )
    parser.add_argument(
        "--camera", default=0,
        help="Camera index (integer) or video URL (string).",
    )
    parser.add_argument(
        "--conf", default=0.25, type=float,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou", default=0.45, type=float,
        help="IOU threshold for non-maximum suppression.",
    )
    parser.add_argument(
        "--width", default=0, type=int,
        help="Requested capture width (0 = camera default).",
    )
    parser.add_argument(
        "--height", default=0, type=int,
        help="Requested capture height (0 = camera default).",
    )
    parser.add_argument(
        "--no-boxes", action="store_true",
        help="Do not draw bounding boxes.",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true",
        help="Do not draw skeleton limbs.",
    )
    return parser.parse_args()


def try_camera_source(camera_arg: str | int) -> int | str:
    """Try to convert the camera argument to an integer index, otherwise keep as string."""
    try:
        return int(camera_arg)
    except (TypeError, ValueError):
        return camera_arg


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print(f"[INFO] Loading model: {args.model}")
    model = PoseEstimationModel(
        args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print("[INFO] Model loaded.")

    # ------------------------------------------------------------------ #
    # Open camera
    # ------------------------------------------------------------------ #
    source = try_camera_source(args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera/source: {source}", file=sys.stderr)
        sys.exit(1)

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win_name = "Pose Estimation – Webcam  |  q/Esc=quit  s=save  p=pause"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    paused = False
    frame_count = 0
    fps_start = time.perf_counter()
    fps = 0.0

    print("[INFO] Stream started. Press 'q' or Esc to quit, 's' to save, 'p' to pause.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame – retrying …")
                time.sleep(0.05)
                continue

            # ---------------------------------------------------------- #
            # Inference
            # ---------------------------------------------------------- #
            detections = model.predict(frame)

            # ---------------------------------------------------------- #
            # Visualisation
            # ---------------------------------------------------------- #
            vis = draw_poses(
                frame.copy(),
                detections,
                draw_boxes=not args.no_boxes,
                draw_skeleton=not args.no_skeleton,
            )

            # FPS counter
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()

            _draw_hud(vis, fps, len(detections))
            display = vis

        # ---------------------------------------------------------- #
        # Display
        # ---------------------------------------------------------- #
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"pose_capture_{ts}.jpg"
            cv2.imwrite(fname, display)
            print(f"[INFO] Saved frame to {fname}")
        elif key == ord("p"):
            paused = not paused
            status = "paused" if paused else "resumed"
            print(f"[INFO] Stream {status}.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


def _draw_hud(
    image, fps: float, num_persons: int, font_scale: float = 0.55
) -> None:
    """Draw a semi-transparent HUD with FPS and person count."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Persons: {num_persons}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    margin = 6
    y = margin

    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        # Background rectangle
        cv2.rectangle(
            image,
            (margin - 2, y),
            (margin + tw + 2, y + th + baseline + 2),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            line,
            (margin, y + th),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += th + baseline + margin


if __name__ == "__main__":
    main()
