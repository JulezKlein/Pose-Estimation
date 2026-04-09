"""Dataset and DataLoader utilities for dart-throw action classification.

Data layout (produced by ``webcam_stream.py --record-dir`` + ``label_tool.py``):

    captures/
        <session>/
            keypoints.jsonl   – one JSON record per frame:
                                 {"frame_idx": int, "timestamp_s": float,
                                  "detections": [{"bbox": [...], "score": float,
                                                  "keypoints": [[x,y,c]×17]}, ...]}
            labels.json       – throw annotations:
                                 {"fps": float, "total_frames": int,
                                  "throws": [{"start_frame": int, "end_frame": int,
                                              "person_idx": int, ...}, ...]}

Each sample is a fixed-length sliding-window sequence of keypoint vectors for
the labeled thrower person.  The label is:
    1  – throw  (the window overlaps a labeled throw interval by ≥ overlap_threshold)
    0  – no-throw / undefined

Input feature vector per frame (input_size = 51):
    17 keypoints × (x, y, confidence) — raw pixel coords, optionally normalised
    by the person's bounding-box width/height so the network is position-invariant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Number of COCO keypoints and values per keypoint (x, y, confidence)
_N_KP = 17
_KP_DIM = 3
INPUT_SIZE = _N_KP * _KP_DIM  # 51


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _load_keypoints_index(jsonl_path: Path) -> dict[int, dict]:
    """Load *all* frames from a keypoints.jsonl into a {frame_idx: record} dict."""
    index: dict[int, dict] = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index[rec["frame_idx"]] = rec
    return index


def _extract_kp_vector(record: dict, person_idx: int) -> np.ndarray:
    """Return a (51,) float32 vector for *person_idx* in *record*.

    If the person is not present in this frame, a zero vector is returned.
    Coordinates are returned as-is (pixel space); normalisation is applied
    later if requested.
    """
    dets = record.get("detections", [])
    if person_idx >= len(dets):
        return np.zeros(INPUT_SIZE, dtype=np.float32)
    kps = dets[person_idx]["keypoints"]  # list of [x, y, conf]
    return np.array(kps, dtype=np.float32).reshape(INPUT_SIZE)


def _bbox_for_person(record: dict, person_idx: int) -> Optional[List[float]]:
    """Return [x1, y1, x2, y2] for *person_idx*, or None if absent."""
    dets = record.get("detections", [])
    if person_idx >= len(dets):
        return None
    return dets[person_idx]["bbox"]


def _normalise_kp_vector(vec: np.ndarray, bbox: Optional[List[float]]) -> np.ndarray:
    """Normalise x/y coordinates relative to the person's bounding box.

    Confidence values are left unchanged.  If bbox is None the raw vector is
    returned unchanged.
    """
    if bbox is None:
        return vec
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    out = vec.copy()
    # indices 0,3,6,… are x; 1,4,7,… are y; 2,5,8,… are confidence
    out[0::3] = (out[0::3] - x1) / bw   # normalised x ∈ [0,1] roughly
    out[1::3] = (out[1::3] - y1) / bh   # normalised y ∈ [0,1] roughly
    return out


def _build_throw_mask(total_frames: int, throws: list[dict]) -> np.ndarray:
    """Return a boolean array of shape (total_frames,) marking throw frames.

    Each element is True if that frame falls within *any* throw interval.
    ``throw["person_idx"]`` is stored alongside for per-window person selection.
    """
    mask = np.zeros(total_frames, dtype=bool)
    for t in throws:
        s, e = int(t["start_frame"]), int(t["end_frame"])
        mask[s : e + 1] = True
    return mask


# --------------------------------------------------------------------------- #
# Per-session window builder
# --------------------------------------------------------------------------- #

def _build_windows(
    kp_index: dict[int, dict],
    total_frames: int,
    throws: list[dict],
    seq_len: int,
    stride: int,
    overlap_threshold: float,
    normalise: bool,
) -> Tuple[List[np.ndarray], List[int]]:
    """Slide a window over the session and collect (sequence, label) pairs.

    For each window we pick the *first* throw that overlaps it (if any) to
    determine which ``person_idx`` to extract keypoints for.  Windows with no
    throw annotation always use person 0 (or the only detected person).

    Parameters
    ----------
    kp_index:
        {frame_idx: record} mapping for the session.
    total_frames:
        Total number of frames in the session.
    throws:
        List of throw dicts from ``labels.json``.
    seq_len:
        Window length in frames.
    stride:
        Step between consecutive windows.
    overlap_threshold:
        Fraction of window frames that must lie within a throw interval for
        the window to be labelled 1.  E.g. 0.5 means ≥50 % overlap.
    normalise:
        Whether to normalise keypoints by the person's bounding box.

    Returns
    -------
    sequences : list of (seq_len, 51) float32 arrays
    labels    : list of int (0 or 1)
    """
    throw_mask = _build_throw_mask(total_frames, throws)

    sequences: List[np.ndarray] = []
    labels: List[int] = []

    for start in range(0, total_frames - seq_len + 1, stride):
        end = start + seq_len  # exclusive

        # Decide label and which person to track
        window_frames = np.arange(start, end)
        overlap = throw_mask[window_frames].mean()
        is_throw = overlap >= overlap_threshold

        # Find the throw (if any) that dominates this window to pick person_idx
        person_idx = 0
        if is_throw:
            # Use person_idx from whichever throw has most overlap with window
            best_overlap = -1.0
            for t in throws:
                t_frames = np.arange(int(t["start_frame"]), int(t["end_frame"]) + 1)
                common = np.intersect1d(window_frames, t_frames).size / seq_len
                if common > best_overlap:
                    best_overlap = common
                    person_idx = int(t.get("person_idx", 0))

        # Build the (seq_len, 51) feature matrix
        seq = np.zeros((seq_len, INPUT_SIZE), dtype=np.float32)
        for i, frame_idx in enumerate(range(start, end)):
            rec = kp_index.get(frame_idx)
            if rec is None:
                continue
            vec = _extract_kp_vector(rec, person_idx)
            if normalise:
                bbox = _bbox_for_person(rec, person_idx)
                vec = _normalise_kp_vector(vec, bbox)
            seq[i] = vec

        sequences.append(seq)
        labels.append(int(is_throw))

    return sequences, labels


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class DartThrowDataset(Dataset):
    """PyTorch Dataset of fixed-length keypoint sequences labelled for dart throws.

    Parameters
    ----------
    captures_dir:
        Root directory containing session sub-folders.
    labels_file:
        Name of the annotation file inside each session folder (default
        ``labels.json``).
    seq_len:
        Number of frames per sequence window.
    stride:
        Frame stride between consecutive windows (default = ``seq_len // 2``
        for 50 % overlap).
    overlap_threshold:
        Fraction of window frames inside a throw interval required to assign
        label 1 (default 0.5).
    normalise:
        Normalise x/y keypoint coordinates by the person's bounding box
        (makes the model position-invariant, default ``True``).
    """

    def __init__(
        self,
        captures_dir: str | Path,
        labels_file: str = "labels.json",
        seq_len: int = 30,
        stride: Optional[int] = None,
        overlap_threshold: float = 0.5,
        normalise: bool = True,
    ) -> None:
        self.captures_dir = Path(captures_dir)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2
        self.overlap_threshold = overlap_threshold
        self.normalise = normalise

        self._sequences: List[np.ndarray] = []
        self._labels: List[int] = []

        self._load_all_sessions(labels_file)

    # ---------------------------------------------------------------------- #

    def _load_all_sessions(self, labels_file: str) -> None:
        sessions = sorted(
            d for d in self.captures_dir.iterdir()
            if d.is_dir() and (d / "keypoints.jsonl").exists()
        )
        if not sessions:
            raise FileNotFoundError(
                f"No sessions with keypoints.jsonl found in {self.captures_dir}"
            )

        for session_dir in sessions:
            lf = session_dir / labels_file
            kp_file = session_dir / "keypoints.jsonl"

            if not lf.exists():
                # Session has no throw labels – all windows get label 0
                throws: list[dict] = []
                meta_file = session_dir / "meta.json"
                if meta_file.exists():
                    total_frames = json.loads(meta_file.read_text())["total_frames"]
                else:
                    # Fall back to counting lines in the JSONL
                    total_frames = sum(1 for _ in open(kp_file, encoding="utf-8"))
            else:
                label_data = json.loads(lf.read_text())
                throws = label_data.get("throws", [])
                total_frames = int(label_data["total_frames"])

            kp_index = _load_keypoints_index(kp_file)

            seqs, lbls = _build_windows(
                kp_index,
                total_frames,
                throws,
                seq_len=self.seq_len,
                stride=self.stride,
                overlap_threshold=self.overlap_threshold,
                normalise=self.normalise,
            )
            self._sequences.extend(seqs)
            self._labels.extend(lbls)

    # ---------------------------------------------------------------------- #
    # Dataset interface
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._sequences[idx])          # (seq_len, 51)
        y = torch.tensor(self._labels[idx], dtype=torch.float32)  # scalar
        return x, y

    # ---------------------------------------------------------------------- #
    # Convenience
    # ---------------------------------------------------------------------- #

    @property
    def class_counts(self) -> dict[int, int]:
        """Return {0: n_no_throw, 1: n_throw} counts."""
        arr = np.array(self._labels)
        return {0: int((arr == 0).sum()), 1: int((arr == 1).sum())}

    def pos_weight(self) -> torch.Tensor:
        """Compute ``pos_weight`` for :class:`torch.nn.BCEWithLogitsLoss`.

        Returns the ratio ``n_negative / n_positive`` as a scalar tensor,
        which compensates for class imbalance.
        """
        counts = self.class_counts
        n_neg = counts.get(0, 1)
        n_pos = counts.get(1, 1)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_dataloaders(
    cfg,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation :class:`~torch.utils.data.DataLoader` objects
    from an OmegaConf config (or any object with ``data``, ``training`` attrs).

    Parameters
    ----------
    cfg:
        OmegaConf DictConfig (or compatible mapping) with at minimum::

            data:
              captures_dir: captures
              labels_file:  labels.json
            training:
              seq_len:    30
              batch_size: 32
              val_split:  0.2
              seed:       42

    num_workers:
        Passed to DataLoader (default 0 = main-process loading).
    pin_memory:
        Passed to DataLoader; set True when training on CUDA.

    Returns
    -------
    train_loader, val_loader
    """
    dataset = DartThrowDataset(
        captures_dir=cfg.data.captures_dir,
        labels_file=cfg.data.labels_file,
        seq_len=cfg.training.seq_len,
        overlap_threshold=cfg.training.overlap_threshold,
        normalise=True,
    )

    counts = dataset.class_counts
    total = len(dataset)
    print(
        f"[DATA] {total} windows  –  "
        f"no-throw: {counts[0]} ({counts[0]/total:.1%})  "
        f"throw: {counts[1]} ({counts[1]/total:.1%})"
    )

    n_val = max(1, int(total * cfg.training.val_split))
    n_train = total - n_val

    generator = torch.Generator().manual_seed(cfg.training.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
