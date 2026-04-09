#!/usr/bin/env python3
"""Video pose-estimation inference.

Reads every frame from an input video file, runs pose estimation, and writes
the annotated output to a new video file.  A progress bar is printed to the
terminal.

Usage
-----
    python3 pose_est_run_video.py --input video.mp4
    python3 pose_est_run_video.py --input video.mp4 --show
    python3 pose_est_run_video.py --input video.mp4 --record-dir captures
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from pose_estimation import PoseEstimationModel, draw_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pose estimation on a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="pose_estimation/model/best.mlpackage", type=Path,
        help="Path to the ONNX (.onnx) or CoreML (.mlpackage / .mlmodel) model.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the input video file.",
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
        "--show", action="store_true",
        help="Display each annotated frame in a window while processing.",
    )
    parser.add_argument(
        "--record-dir", default="captures", type=Path,
        help=(
            "Root directory for training-data capture. "
            "Creates a timestamped sub-folder containing "
            "'keypoints.jsonl' (one JSON object per frame) and 'meta.json'."
        ),
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

    # ------------------------------------------------------------------ #
    # Set up session dir (always) and resolve output path
    # ------------------------------------------------------------------ #
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.record_dir / ts
    session_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.input.suffix or ".mp4"
    output_path = session_dir / f"{args.input.stem}_pose{suffix}"

    print(f"[INFO] Session → {session_dir}")
    print(
        f"[INFO] Video: {width}×{height}, {fps:.2f} fps, "
        f"~{total_frames} frames  →  {output_path}"
    )

    # ------------------------------------------------------------------ #
    # Set up writers
    # ------------------------------------------------------------------ #
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {output_path}", file=sys.stderr)
        sys.exit(1)

    kp_file = open(session_dir / "keypoints.jsonl", "w", encoding="utf-8")

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
        )

        writer.write(vis)

        record = {
            "frame_idx": frame_idx,
            "timestamp_s": round(frame_idx / fps, 4),
            "detections": [
                {
                    "bbox": [round(float(v), 2) for v in d["box"]],
                    "score": round(float(d["score"]), 4),
                    "keypoints": [
                        [round(float(x), 2), round(float(y), 2), round(float(c), 3)]
                        for x, y, c in d["keypoints"]
                    ],
                }
                for d in detections
            ],
        }
        kp_file.write(json.dumps(record) + "\n")

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
    kp_file.close()
    meta = {
        "fps": fps,
        "total_frames": frame_idx,
        "source": str(args.input),
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Session saved → {session_dir}  ({frame_idx} frames)")
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(
        f"\n[INFO] Processed {frame_idx} frames in {elapsed:.1f}s "
        f"({frame_idx / elapsed:.1f} fps)."
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
