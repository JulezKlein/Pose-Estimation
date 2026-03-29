"""ONNX / CoreML pose-estimation model wrapper.

Supports YOLO26 pose models that accept 640×640 RGB images and export with
an integrated end-to-end head.

Output tensor layout (YOLO26 end2end export)
---------------------------------------------
Shape: [1, 300, 57]
    - [:, :, 0:4]   – bounding-box (x1, y1, x2, y2) in input-pixel space
    - [:, :, 4]     – detection confidence score (0–1)
    - [:, :, 5]     – class ID
    - [:, :, 6:57]  – 17 keypoints, each encoded as (x, y, visibility)
                                        in input-pixel space  →  51 values total
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Type alias
# --------------------------------------------------------------------------- #
Detection = dict  # keys: box, score, class_id, keypoints

INPUT_SIZE = 640  # model input width == height


class PoseEstimationModel:
    """Load and run a pose-estimation model (ONNX or CoreML).

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` or ``.mlpackage`` / ``.mlmodel`` file.
    conf_threshold:
        Minimum detection confidence to keep (default 0.25).
    iou_threshold:
        Kept for API compatibility. Not used for YOLO26 end2end outputs,
        because NMS is already performed in-model.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self._session = None
        self._coreml_model = None
        self._backend: str = ""

        self._load_model()

    # ---------------------------------------------------------------------- #
    # Model loading
    # ---------------------------------------------------------------------- #

    def _load_model(self) -> None:
        suffix = self.model_path.suffix.lower()

        if suffix == ".onnx":
            self._load_onnx()
        elif suffix in {".mlpackage", ".mlmodel", ""}:
            self._load_coreml()
        else:
            raise ValueError(
                f"Unsupported model format '{suffix}'. "
                "Use '.onnx', '.mlpackage', or '.mlmodel'."
            )

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with: pip install onnxruntime"
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self.model_path), providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._backend = "onnx"

    def _load_coreml(self) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError(
                "CoreML inference is only supported on macOS."
            )
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "coremltools is required for CoreML inference. "
                "Install it with: pip install coremltools"
            ) from exc

        self._coreml_model = ct.models.MLModel(str(self.model_path))
        self._backend = "coreml"

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def predict(self, image_bgr: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR image.

        Parameters
        ----------
        image_bgr:
            OpenCV BGR image (H × W × 3, uint8).

        Returns
        -------
        List of detections, each a dict::

            {
                "box":       [x1, y1, x2, y2],   # pixel coords in *original* image
                "score":     float,
                "keypoints": np.ndarray,          # shape (17, 3) – x, y, visibility
            }
        """
        orig_h, orig_w = image_bgr.shape[:2]

        # Pre-process
        blob, ratio, (pad_w, pad_h) = self._letterbox(image_bgr)

        # Inference
        raw = self._run_inference(blob)

        # Post-process
        detections = self._postprocess(raw, orig_w, orig_h, ratio, pad_w, pad_h)

        return detections

    # ---------------------------------------------------------------------- #
    # Pre-processing
    # ---------------------------------------------------------------------- #

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize *image* to INPUT_SIZE×INPUT_SIZE with grey padding (letterbox).

        Returns
        -------
        blob:
            Float32 array with shape (1, 3, INPUT_SIZE, INPUT_SIZE), values in [0, 1].
        ratio:
            Scale factor applied to both axes.
        (pad_w, pad_h):
            Total horizontal / vertical padding added (in pixels).
        """
        h, w = image.shape[:2]
        ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)

        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding to reach INPUT_SIZE × INPUT_SIZE
        pad_w = INPUT_SIZE - new_w
        pad_h = INPUT_SIZE - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # BGR → RGB, HWC → CHW, normalise to [0, 1]
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        return blob, ratio, (pad_w, pad_h)

    # ---------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------- #

    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        if self._backend == "onnx":
            outputs = self._session.run(None, {self._input_name: blob})
            return outputs[0]  # expected: (1, 300, 57)
        elif self._backend == "coreml":
            return self._run_coreml(blob)
        raise RuntimeError(f"Unknown backend '{self._backend}'.")

    def _run_coreml(self, blob: np.ndarray) -> np.ndarray:
        from PIL import Image

        input_name = list(self._coreml_model.get_spec().description.input)[0].name

        # CoreML Image inputs expect a PIL Image (uint8 RGB HWC).
        # blob is (1, 3, H, W) float32 [0, 1] RGB — convert back to uint8.
        img_hwc = (blob[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_hwc)

        result = self._coreml_model.predict({input_name: pil_image})
        output = list(result.values())[0]

        # Normalize CoreML outputs to (1, N, 57) for end2end decoding.
        if output.ndim == 2:
            output = output[np.newaxis, ...]
        if output.ndim != 3:
            raise ValueError(f"Unexpected CoreML output rank: {output.ndim}")
        if output.shape[-1] == 57:
            return output
        if output.shape[1] == 57:
            return output.transpose(0, 2, 1)

        raise ValueError(
            f"Unexpected CoreML output shape {output.shape}; expected (1, N, 57) "
            "or (1, 57, N)."
        )

    # ---------------------------------------------------------------------- #
    # Post-processing
    # ---------------------------------------------------------------------- #

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        ratio: float,
        pad_w: float,
        pad_h: float,
    ) -> List[Detection]:
        """Decode raw model output into a list of detections.

        *raw* is expected to have shape (1, N, 57), where each detection row is:
        [x1, y1, x2, y2, score, class_id, 17*3 keypoints].
        """
        if raw.ndim != 3:
            raise ValueError(f"Unexpected model output rank: {raw.ndim}")

        if raw.shape[-1] == 57:
            preds = raw[0]  # (N, 57)
        elif raw.shape[1] == 57:
            # Compatibility with transposed exports.
            preds = raw[0].T
        else:
            raise ValueError(
                f"Unexpected model output shape {raw.shape}; expected (1, N, 57) "
                "or (1, 57, N)."
            )

        scores = preds[:, 4]
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []

        preds = preds[mask]
        scores = scores[mask]

        # End-to-end head already returns final xyxy detections.
        boxes_lb = preds[:, 0:4].copy()

        # Map back from letterboxed space to original image space
        half_pad_w = pad_w / 2
        half_pad_h = pad_h / 2

        detections: List[Detection] = []
        for i in range(len(preds)):
            bx1, by1, bx2, by2 = boxes_lb[i]
            # Remove padding and rescale
            bx1 = (bx1 - half_pad_w) / ratio
            by1 = (by1 - half_pad_h) / ratio
            bx2 = (bx2 - half_pad_w) / ratio
            by2 = (by2 - half_pad_h) / ratio

            # Clamp to image boundaries
            bx1 = float(np.clip(bx1, 0, orig_w))
            by1 = float(np.clip(by1, 0, orig_h))
            bx2 = float(np.clip(bx2, 0, orig_w))
            by2 = float(np.clip(by2, 0, orig_h))

            # Keypoints: 17 × 3 (x, y, visibility) in letterboxed space
            kps_raw = preds[i, 6:57].reshape(17, 3).copy()
            kps_raw[:, 0] = (kps_raw[:, 0] - half_pad_w) / ratio
            kps_raw[:, 1] = (kps_raw[:, 1] - half_pad_h) / ratio
            kps_raw[:, 0] = np.clip(kps_raw[:, 0], 0, orig_w)
            kps_raw[:, 1] = np.clip(kps_raw[:, 1], 0, orig_h)

            detections.append(
                {
                    "box": [bx1, by1, bx2, by2],
                    "score": float(scores[i]),
                    "class_id": int(preds[i, 5]),
                    "keypoints": kps_raw,  # (17, 3) float32
                }
            )

        return detections

    # ---------------------------------------------------------------------- #
    # Note
    # ---------------------------------------------------------------------- #

    # The YOLO26 end2end export performs NMS in-model, so no external NMS
    # is needed in this wrapper.
