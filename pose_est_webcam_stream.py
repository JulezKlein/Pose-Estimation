#!/usr/bin/env python3
"""Webcam pose-estimation stream.

Captures the default webcam (or any OpenCV-compatible camera index / URL),
runs pose estimation on every frame, and shows the result in a window with
skeleton overlay.

Usage
-----
    python pose_est_webcam_stream.py --model path/to/model.onnx
    python pose_est_webcam_stream.py --model path/to/model.onnx --camera 1
    python pose_est_webcam_stream.py --model path/to/model.onnx --conf 0.35 --iou 0.5
    python pose_est_webcam_stream.py --model path/to/model.onnx --save-video out.mp4
    python pose_est_webcam_stream.py --model path/to/model.onnx --record-dir captures

Controls
--------
    q / Esc  – quit
    s        – save current frame to disk
    p        – pause / resume
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, List, Optional

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
        help=(
            "Retained for compatibility; ignored for YOLO26 end-to-end "
            "models (NMS is in-model)."
        ),
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
    parser.add_argument(
        "--save-video", default=None, type=Path,
        help="Optional output path to save the annotated webcam stream (e.g. out.mp4).",
    )
    parser.add_argument(
        "--record-dir", default=None, type=Path,
        help=(
            "Root directory for training-data capture sessions. "
            "Each recording creates a timestamped sub-folder containing "
            "'raw.mp4' (unannotated video) and 'keypoints.jsonl' "
            "(one JSON object per frame). "
            "Recording starts automatically when this flag is set; "
            "press 'r' to stop/start additional segments."
        ),
    )
    return parser.parse_args()


def try_camera_source(camera_arg: str | int) -> int | str:
    """Try to convert the camera argument to an integer index, otherwise keep as string."""
    try:
        return int(camera_arg)
    except (TypeError, ValueError):
        return camera_arg


# --------------------------------------------------------------------------- #
# Recording session
# --------------------------------------------------------------------------- #

@dataclass
class RecordingSession:
    """Manages a single capture segment: raw video + per-frame keypoints."""

    session_dir: Path
    raw_writer: cv2.VideoWriter
    kp_file: IO
    fps: float
    frame_count: int = 0

    def write(self, raw_frame: "cv2.Mat", detections: list, timestamp_s: float) -> None:
        """Write one raw frame and its detections to disk."""
        self.raw_writer.write(raw_frame)
        record = {
            "frame_idx": self.frame_count,
            "timestamp_s": round(timestamp_s, 4),
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
        self.kp_file.write(json.dumps(record) + "\n")
        self.frame_count += 1

    def close(self) -> None:
        self.raw_writer.release()
        self.kp_file.close()
        # Write small metadata sidecar
        meta = {
            "fps": self.fps,
            "total_frames": self.frame_count,
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[INFO] Recording saved → {self.session_dir}  ({self.frame_count} frames)")


def start_recording(
    session_root: Path,
    cap: cv2.VideoCapture,
    frame_w: int,
    frame_h: int,
) -> Optional[RecordingSession]:
    """Create a new timestamped session folder and open writers."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = session_root / ts
    session_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(
        str(session_dir / "raw.mp4"), fourcc, fps, (frame_w, frame_h)
    )
    if not raw_writer.isOpened():
        print(f"[ERROR] Cannot open raw video writer in {session_dir}", file=sys.stderr)
        return None

    kp_file = open(session_dir / "keypoints.jsonl", "w", encoding="utf-8")
    print(f"[INFO] Recording started → {session_dir}")
    return RecordingSession(
        session_dir=session_dir,
        raw_writer=raw_writer,
        kp_file=kp_file,
        fps=fps,
    )


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

    win_name = "Pose Estimation – Webcam  |  q/Esc=quit  s=save  p=pause  r=record"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    paused = False
    frame_count = 0
    fps_start = time.perf_counter()
    fps = 0.0
    display = None
    writer = None
    video_out_path: Path | None = None

    # Recording state
    recording: Optional[RecordingSession] = None
    rec_start_time: float = 0.0
    stream_start_time: float = time.perf_counter()

    if args.record_dir is not None:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recording = start_recording(args.record_dir, cap, frame_w, frame_h)
        rec_start_time = time.perf_counter()

    print(
        "[INFO] Stream started. "
        "Press 'q'/Esc to quit, 's' to save frame, 'p' to pause, 'r' to toggle recording."
    )

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

            # ---------------------------------------------------------- #
            # Capture raw frame + keypoints
            # ---------------------------------------------------------- #
            if recording is not None:
                timestamp_s = time.perf_counter() - rec_start_time
                recording.write(frame, detections, timestamp_s)

            _draw_hud(vis, fps, len(detections), recording=recording)
            display = vis

            if args.save_video is not None:
                if writer is None:
                    video_out_path = args.save_video
                    video_out_path.parent.mkdir(parents=True, exist_ok=True)

                    out_h, out_w = display.shape[:2]
                    out_fps = cap.get(cv2.CAP_PROP_FPS)
                    if out_fps <= 0:
                        out_fps = 30.0

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(video_out_path), fourcc, out_fps, (out_w, out_h)
                    )
                    if not writer.isOpened():
                        print(
                            f"[ERROR] Cannot create output video: {video_out_path}",
                            file=sys.stderr,
                        )
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(1)
                    print(f"[INFO] Recording stream to: {video_out_path}")

                writer.write(display)

        # ---------------------------------------------------------- #
        # Display
        # ---------------------------------------------------------- #
        if display is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
                status = "paused" if paused else "resumed"
                print(f"[INFO] Stream {status}.")
            continue

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
        elif key == ord("r"):
            if recording is None:
                if args.record_dir is not None:
                    frame_w = display.shape[1]
                    frame_h = display.shape[0]
                    recording = start_recording(args.record_dir, cap, frame_w, frame_h)
                    rec_start_time = time.perf_counter()
                else:
                    print(
                        "[WARN] No --record-dir specified. "
                        "Restart with --record-dir <path> to enable recording."
                    )
            else:
                recording.close()
                recording = None
                print("[INFO] Recording stopped.")

    if recording is not None:
        recording.close()
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Saved video to: {video_out_path}")
    cv2.destroyAllWindows()
    print("[INFO] Done.")


def _draw_hud(
    image,
    fps: float,
    num_persons: int,
    font_scale: float = 0.55,
    recording: Optional["RecordingSession"] = None,
) -> None:
    """Draw a semi-transparent HUD with FPS, person count, and recording indicator."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Persons: {num_persons}",
    ]
    if recording is not None:
        lines.append(f"REC  {recording.frame_count} fr")
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    margin = 6
    y = margin

    for i, line in enumerate(lines):
        is_rec = recording is not None and i == len(lines) - 1
        color = (0, 0, 220) if is_rec else (0, 255, 0)
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        # Background rectangle
        cv2.rectangle(
            image,
            (margin - 2, y),
            (margin + tw + 2, y + th + baseline + 2),
            (0, 0, 0),
            cv2.FILLED,
        )
        # Red dot before REC line
        if is_rec:
            dot_r = max(4, th // 2)
            cx = margin + tw + 2 + dot_r + 4
            cy = y + th // 2
            cv2.circle(image, (cx, cy), dot_r, (0, 0, 220), cv2.FILLED)
        cv2.putText(
            image,
            line,
            (margin, y + th),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += th + baseline + margin


if __name__ == "__main__":
    main()
