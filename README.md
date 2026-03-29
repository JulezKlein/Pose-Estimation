# Pose-Estimation

Human pose estimation with 17 COCO keypoints using an ONNX or CoreML model
(640 × 640 input, YOLO26 end-to-end output compatible).  Three ready-to-run applications
are provided:

| Script | Description |
|---|---|
| `webcam_stream.py` | Live webcam feed with real-time pose overlay |
| `run_video.py` | Batch inference on a video file |
| `run_image.py` | Inference on one or more image files |

---

## Quick start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS / CoreML** – uncomment the `coremltools` line in `requirements.txt`
> and use a `.mlpackage` or `.mlmodel` file as the model path.

### 2 – Obtain a model

Export a YOLO pose model to ONNX with end-to-end decoding:

```bash
pip install ultralytics
yolo export model=yolo26s-pose.pt format=onnx imgsz=640 end2end=True
# produces yolo26s-pose.onnx
```

Or download a pre-exported file from the
[Ultralytics releases](https://github.com/ultralytics/assets/releases).

---

## Usage

### Webcam stream

```bash
python webcam_stream.py --model yolo26s-pose.onnx
```

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `s` | Save current frame as JPEG |
| `p` | Pause / resume |

Optional arguments:
```
--camera 1          # camera index or RTSP/HTTP URL
--conf 0.35         # detection confidence threshold  (default 0.25)
--iou  0.5          # compatibility flag (ignored in end2end mode)
--save-video out.mp4 # save annotated webcam stream to a video file
--width / --height  # request capture resolution
--no-boxes          # hide bounding boxes
--no-skeleton       # hide skeleton limbs
```

---

### Video inference

```bash
python run_video.py --model yolo26s-pose.onnx --input clip.mp4
```

The annotated video is saved as `clip_pose.mp4` by default.

```
--output result.mp4   # custom output path
--show                # preview each frame while processing
--conf / --iou        # thresholds (same as above)
--no-boxes / --no-skeleton
```

---

### Image inference

```bash
python run_image.py --model yolo26s-pose.onnx --input photo.jpg
```

The annotated image is saved as `photo_pose.jpg` by default.

```bash
# Multiple images
python run_image.py --model yolo26s-pose.onnx --input *.jpg --show

# Custom output path (single image only)
python run_image.py --model yolo26s-pose.onnx --input photo.jpg --output out.jpg
```

```
--show             # display result in a window (any key = next, q/Esc = quit)
--no-save          # do not write output files
--conf / --iou     # confidence + compatibility flag (iou ignored in end2end)
--no-boxes / --no-skeleton
```

---

## End-to-end output format

This project assumes the default YOLO26 end-to-end output tensor:

```
shape: (batch_size, 300, 57)

0:4    -> bounding box in xyxy format
4      -> confidence score
5      -> class id
6:57   -> 17 keypoints * 3 values [x, y, visibility]
```

---

## Keypoints (COCO 17)

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

---

## Project structure

```
.
├── pose_estimation/
│   ├── __init__.py        # package exports
│   ├── model.py           # ONNX / CoreML model wrapper + pre/post-processing
│   └── visualization.py   # keypoint & skeleton drawing utilities
├── webcam_stream.py       # real-time webcam application
├── run_video.py           # video inference application
├── run_image.py           # image inference application
└── requirements.txt
```
