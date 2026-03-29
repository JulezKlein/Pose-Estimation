#!/usr/bin/env python3
"""Single-image pose-estimation inference.

Loads one or more image files, runs pose estimation on each, saves the
annotated result, and optionally displays it.

Usage
-----
    python run_image.py --model path/to/model.onnx --input photo.jpg
    python run_image.py --model path/to/model.onnx --input *.jpg --show
    python run_image.py --model path/to/model.onnx --input photo.jpg \\
                        --output result.jpg --conf 0.3
"""

import argparse
import sys
from pathlib import Path

import cv2

from pose_estimation import PoseEstimationModel, draw_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pose estimation on image(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Path to the ONNX (.onnx) or CoreML (.mlpackage / .mlmodel) model.",
    )
    parser.add_argument(
        "--input", required=True, nargs="+", type=Path,
        help="One or more input image files.",
    )
    parser.add_argument(
        "--output", default=None, type=Path,
        help=(
            "Output path for annotated image (single input only). "
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
        help="Display each annotated image in a window.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save the annotated image(s) to disk.",
    )
    return parser.parse_args()


def _output_path(input_path: Path, output_arg: Path | None) -> Path:
    if output_arg is not None:
        return output_arg
    return input_path.parent / f"{input_path.stem}_pose{input_path.suffix}"


def main() -> None:
    args = parse_args()

    if args.output is not None and len(args.input) > 1:
        print(
            "[ERROR] --output can only be used with a single --input file.",
            file=sys.stderr,
        )
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

    if args.show:
        win_name = "Pose Estimation – Image  |  any key=next  q/Esc=quit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------ #
    # Process each image
    # ------------------------------------------------------------------ #
    for img_path in args.input:
        if not img_path.exists():
            print(f"[WARN] File not found, skipping: {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Cannot read image, skipping: {img_path}")
            continue

        h, w = frame.shape[:2]

        # Inference
        detections = model.predict(frame)
        print(
            f"[INFO] {img_path.name}: {w}×{h}  →  "
            f"{len(detections)} person(s) detected."
        )
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            print(
                f"       Person {i+1}: box=({x1},{y1},{x2},{y2})  "
                f"score={det['score']:.3f}"
            )

        # Visualise
        vis = draw_poses(
            frame.copy(),
            detections,
            draw_boxes=not args.no_boxes,
            draw_skeleton=not args.no_skeleton,
        )

        # Save
        if not args.no_save:
            out_path = _output_path(img_path, args.output)
            cv2.imwrite(str(out_path), vis)
            print(f"[INFO] Annotated image saved to: {out_path}")

        # Display
        if args.show:
            _fit_window(win_name, vis)
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Quit.")
                break

    if args.show:
        cv2.destroyAllWindows()

    print("[INFO] Done.")


def _fit_window(win_name: str, image, max_dim: int = 1200) -> None:
    """Resize window so the image fits on screen (up to *max_dim* pixels)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    cv2.resizeWindow(win_name, int(w * scale), int(h * scale))


if __name__ == "__main__":
    main()
