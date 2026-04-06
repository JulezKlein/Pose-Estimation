#!/usr/bin/env python3
"""Video pose-estimation inference.

Reads every frame from an input video file, runs pose estimation, and writes
the annotated output to a new video file.  A progress bar is printed to the
terminal.

Usage
-----
    python pose_est_run_video.py --model path/to/model.onnx --input video.mp4
    python pose_est_run_video.py --model path/to/model.onnx --input video.mp4 \\
                        --output annotated.mp4 --conf 0.35
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from pose_estimation import PoseEstimationModel, draw_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pose estimation on a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Path to the ONNX (.onnx) or CoreML (.mlpackage / .mlmodel) model.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", default=None, type=Path,
        help=(
            "Path for the annotated output video. "
            "Defaults to <input_stem>_pose.<input_suffix>."
        ),
    )
    parser.add_argument(
        "--conf", default=0.25, type=float,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou", default=0.45, type=float,
        help=(
            "Retained for compatibility; ignored for YOLO26 end-to-end "
            "models (NMS is in-model)."
        ),
    )
    parser.add_argument(
        "--no-boxes", action="store_true",
        help="Do not draw bounding boxes.",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true",
        help="Do not draw skeleton limbs.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display each annotated frame in a window while processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Validate input
    # ------------------------------------------------------------------ #
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Resolve output path
    # ------------------------------------------------------------------ #
    if args.output is None:
        suffix = args.input.suffix or ".mp4"
        args.output = args.input.parent / f"{args.input.stem}_pose{suffix}"

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
    # Open video
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[INFO] Video: {width}×{height}, {fps:.2f} fps, "
        f"~{total_frames} frames  →  {args.output}"
    )

    # ------------------------------------------------------------------ #
    # Set up writer
    # ------------------------------------------------------------------ #
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {args.output}", file=sys.stderr)
        sys.exit(1)

    if args.show:
        win_name = "Pose Estimation – Video  |  q/Esc=quit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------ #
    # Process frames
    # ------------------------------------------------------------------ #
    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model.predict(frame)
        vis = draw_poses(
            frame,
            detections,
            draw_boxes=not args.no_boxes,
            draw_skeleton=not args.no_skeleton,
        )

        writer.write(vis)
        frame_idx += 1

        # Progress
        _print_progress(frame_idx, total_frames, t_start)

        if args.show:
            cv2.imshow(win_name, vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n[INFO] Interrupted by user.")
                break

    # ------------------------------------------------------------------ #
    # Clean up
    # ------------------------------------------------------------------ #
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(
        f"\n[INFO] Processed {frame_idx} frames in {elapsed:.1f}s "
        f"({frame_idx / elapsed:.1f} fps). "
        f"Output saved to: {args.output}"
    )


def _print_progress(current: int, total: int, t_start: float) -> None:
    """Print a compact one-line progress bar to stdout."""
    elapsed = time.perf_counter() - t_start
    speed = current / elapsed if elapsed > 0 else 0
    if total > 0:
        pct = current / total * 100
        eta = (total - current) / speed if speed > 0 else 0
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        msg = (
            f"\r[{bar}] {pct:5.1f}%  "
            f"frame {current}/{total}  "
            f"{speed:.1f} fps  "
            f"ETA {eta:.0f}s   "
        )
    else:
        msg = f"\rFrame {current}  {speed:.1f} fps   "

    sys.stdout.write(msg)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
