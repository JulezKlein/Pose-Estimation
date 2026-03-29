"""Pose visualisation utilities.

Draws bounding boxes, COCO 17-keypoints, and the human-body skeleton on
OpenCV BGR images.

COCO keypoint index → body part
--------------------------------
 0  nose
 1  left_eye
 2  right_eye
 3  left_ear
 4  right_ear
 5  left_shoulder
 6  right_shoulder
 7  left_elbow
 8  right_elbow
 9  left_wrist
10  right_wrist
11  left_hip
12  right_hip
13  left_knee
14  right_knee
15  left_ankle
16  right_ankle
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

# Colour palette (BGR) – one colour per person (cycling)
_PERSON_COLORS = [
    (255, 56,  56),
    ( 56, 255,  56),
    ( 56,  56, 255),
    (255, 255,  56),
    (255,  56, 255),
    ( 56, 255, 255),
    (255, 128,   0),
    (128,   0, 255),
    (  0, 128, 255),
    (255,   0, 128),
]

# Skeleton edges: (from_idx, to_idx)
_SKELETON = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Shoulders
    (5, 6),
    # Torso
    (5, 11), (6, 12),
    # Hips
    (11, 12),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

# Per-joint colours (BGR) matching left/right symmetry
_JOINT_COLORS = [
    (255, 255,   0),   # 0  nose          – yellow
    (255, 128,   0),   # 1  left_eye      – orange
    (  0, 255, 128),   # 2  right_eye     – cyan-green
    (255,  64,   0),   # 3  left_ear      – deep orange
    (  0, 255,  64),   # 4  right_ear     – lime
    (255,   0,   0),   # 5  left_shoulder – blue (B=255)
    (  0,   0, 255),   # 6  right_shoulder – red (R=255)
    (255,  64,  64),   # 7  left_elbow    – light blue
    ( 64,  64, 255),   # 8  right_elbow   – light red
    (255, 128, 128),   # 9  left_wrist    – pale blue
    (128, 128, 255),   # 10 right_wrist   – pale red
    (  0, 255,   0),   # 11 left_hip      – green
    (  0, 128,   0),   # 12 right_hip     – dark green
    ( 64, 255,  64),   # 13 left_knee     – light green
    ( 32, 200,  32),   # 14 right_knee    – medium green
    ( 64, 255, 128),   # 15 left_ankle    – teal
    ( 32, 200,  96),   # 16 right_ankle   – dark teal
]

# Visibility threshold – keypoints with confidence below this are not drawn
_VIS_THRESHOLD = 0.3

# Keypoint names (index → string)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def draw_poses(
    image: np.ndarray,
    detections: List[dict],
    draw_boxes: bool = True,
    draw_keypoints: bool = True,
    draw_skeleton: bool = True,
    keypoint_radius: int = 5,
    skeleton_thickness: int = 2,
    box_thickness: int = 2,
    vis_threshold: float = _VIS_THRESHOLD,
) -> np.ndarray:
    """Draw all detections onto *image* (in-place) and return it.

    Parameters
    ----------
    image:
        BGR OpenCV image (H × W × 3, uint8).  Modified in-place.
    detections:
        List of dicts as returned by :meth:`PoseEstimationModel.predict`.
    draw_boxes:
        Whether to draw the bounding box for each person.
    draw_keypoints:
        Whether to draw coloured circles at each keypoint.
    draw_skeleton:
        Whether to draw skeleton limb lines.
    keypoint_radius:
        Radius of keypoint circles in pixels.
    skeleton_thickness:
        Thickness of skeleton limb lines.
    box_thickness:
        Thickness of bounding-box rectangles.
    vis_threshold:
        Minimum keypoint visibility score to draw the keypoint / limb.

    Returns
    -------
    The same *image* with visualisation painted on it.
    """
    for idx, det in enumerate(detections):
        color = _PERSON_COLORS[idx % len(_PERSON_COLORS)]

        # ------------------------------------------------------------------ #
        # Bounding box
        # ------------------------------------------------------------------ #
        if draw_boxes:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
            label = f"{det['score']:.2f}"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image,
                (x1, y1 - lh - baseline - 4),
                (x1 + lw + 4, y1),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        keypoints = det.get("keypoints")
        if keypoints is None:
            continue
        # keypoints: (17, 3) – x, y, visibility

        # ------------------------------------------------------------------ #
        # Skeleton
        # ------------------------------------------------------------------ #
        if draw_skeleton:
            for kp_a, kp_b in _SKELETON:
                xa, ya, va = keypoints[kp_a]
                xb, yb, vb = keypoints[kp_b]
                if va < vis_threshold or vb < vis_threshold:
                    continue
                pt_a = (int(xa), int(ya))
                pt_b = (int(xb), int(yb))
                limb_color = _blend_colors(
                    _JOINT_COLORS[kp_a], _JOINT_COLORS[kp_b]
                )
                cv2.line(image, pt_a, pt_b, limb_color, skeleton_thickness, cv2.LINE_AA)

        # ------------------------------------------------------------------ #
        # Keypoints
        # ------------------------------------------------------------------ #
        if draw_keypoints:
            for kp_idx, (x, y, v) in enumerate(keypoints):
                if v < vis_threshold:
                    continue
                jcolor = _JOINT_COLORS[kp_idx]
                cv2.circle(
                    image,
                    (int(x), int(y)),
                    keypoint_radius,
                    jcolor,
                    cv2.FILLED,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    image,
                    (int(x), int(y)),
                    keypoint_radius,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    return image


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _blend_colors(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the midpoint colour between *c1* and *c2* (BGR)."""
    return (
        (c1[0] + c2[0]) // 2,
        (c1[1] + c2[1]) // 2,
        (c1[2] + c2[2]) // 2,
    )
