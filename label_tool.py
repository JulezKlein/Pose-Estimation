#!/usr/bin/env python3
"""Dart-throw labeling tool (Streamlit).

Replay a recorded session frame-by-frame, mark the start and end of each
dart throw, and save the labels as JSON alongside the recording.

Usage
-----
    streamlit run label_tool.py
    streamlit run label_tool.py --captures-dir path/to/captures

The captures directory is expected to contain sub-folders produced by
``webcam_stream.py --record-dir <captures-dir>``.  Each sub-folder holds:

    raw.mp4 / *.MOV  – unannotated video
    keypoints.jsonl  – per-frame pose detections (one JSON object per line)
    meta.json        – fps + total_frames metadata
    labels.json      – (created/updated by this tool)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from pose_estimation import draw_poses

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _find_video(session_dir: Path) -> Optional[Path]:
    """Return the first video file found in *session_dir*, or None."""
    for p in sorted(session_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            return p
    return None

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Dart Throw Labeler",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# CLI args (passed after -- when launching with `streamlit run … -- …`)
# --------------------------------------------------------------------------- #
@st.cache_data
def _get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--captures-dir", default="captures", type=Path)
    args, _ = parser.parse_known_args()
    return args


cli = _get_cli_args()
CAPTURES_DIR: Path = cli.captures_dir

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    m, s_int = divmod(int(seconds), 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{m:02d}:{s_int:02d}.{ms:03d}"


@st.cache_resource
def _load_video_meta(video_path: str) -> dict:
    """Read video properties without decoding frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "total_frames": total, "width": width, "height": height}


def _read_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Seek to *frame_idx* and return the frame as an RGB numpy array."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def _load_keypoints_for_frame(kp_path: Path, frame_idx: int) -> Optional[dict]:
    """Return the keypoints record for *frame_idx* by scanning the JSONL file.

    Linear scan is acceptable for labeling use; the file is not large enough
    to warrant an index.
    """
    if not kp_path.exists():
        return None
    try:
        with open(kp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("frame_idx") == frame_idx:
                        return rec
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return None


def _overlay_keypoints(
    image: np.ndarray,
    kp_record: dict,
    highlight_idx: Optional[int] = None,
) -> np.ndarray:
    """Draw detected skeletons on *image* (RGB) using the shared visualization utils.

    If *highlight_idx* is given, a distinct bounding box and "Thrower" label
    are drawn around that detection so the selected person is clearly visible.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = [
        {
            "box": det["bbox"],
            "score": det["score"],
            "keypoints": np.array(det["keypoints"], dtype=np.float32),
        }
        for det in kp_record.get("detections", [])
    ]
    draw_poses(bgr, detections)
    # Highlight the selected thrower with a cyan bounding box + label
    if highlight_idx is not None and highlight_idx < len(detections):
        x1, y1, x2, y2 = [int(v) for v in detections[highlight_idx]["box"]]
        cv2.rectangle(bgr, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 3)
        label = f"Thrower (Person {highlight_idx})"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            bgr,
            (x1 - 3, y1 - lh - baseline - 10),
            (x1 + lw + 4, y1 - 3),
            (0, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            bgr, label, (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Session discovery
# --------------------------------------------------------------------------- #
if not CAPTURES_DIR.exists():
    st.error(f"Captures directory not found: `{CAPTURES_DIR}`")
    st.info(
        "Run `webcam_stream.py --record-dir captures/` first to create sessions."
    )
    st.stop()

sessions = sorted(
    [d for d in CAPTURES_DIR.iterdir() if d.is_dir() and _find_video(d) is not None],
    reverse=True,
)
if not sessions:
    st.error("No sessions found in captures directory.")
    st.info("Run `webcam_stream.py --record-dir captures/` to record first.")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar – session picker
# --------------------------------------------------------------------------- #
st.sidebar.title("🎯 Dart Throw Labeler")
st.sidebar.markdown("---")

session_names = [s.name for s in sessions]
selected_name = st.sidebar.selectbox("Session", session_names)

session_dir = CAPTURES_DIR / selected_name
_video_file = _find_video(session_dir)
if _video_file is None:
    st.error(f"No video file found in session: `{session_dir}`")
    st.stop()
video_path = str(_video_file)
kp_path = session_dir / "keypoints.jsonl"
labels_path = session_dir / "labels.json"

# Reset state when the session changes
if st.session_state.get("_loaded_session") != selected_name:
    st.session_state["_loaded_session"] = selected_name
    st.session_state["throws"] = []
    st.session_state["pending_start"] = None
    st.session_state["pending_person_idx"] = None
    st.session_state["selected_person_idx"] = 0
    st.session_state["frame_idx"] = 0
    st.session_state["show_skeleton"] = True
    # Load existing labels if present
    if labels_path.exists():
        try:
            data = json.loads(labels_path.read_text())
            st.session_state["throws"] = data.get("throws", [])
        except Exception:
            pass

# --------------------------------------------------------------------------- #
# Video metadata
# --------------------------------------------------------------------------- #
meta = _load_video_meta(video_path)
fps: float = meta["fps"]
total_frames: int = meta["total_frames"]

st.sidebar.markdown("**Session info**")
st.sidebar.caption(
    f"Frames: {total_frames}  \n"
    f"FPS: {fps:.1f}  \n"
    f"Duration: {_fmt_time(total_frames / fps)}  \n"
    f"Throws labeled: {len(st.session_state['throws'])}"
)
st.sidebar.markdown("---")
show_skeleton = st.sidebar.toggle("Show skeleton overlay", value=st.session_state["show_skeleton"])
st.session_state["show_skeleton"] = show_skeleton

# --------------------------------------------------------------------------- #
# Main layout – tabs: Play | Label
# --------------------------------------------------------------------------- #
st.title(f"Session: {selected_name}")

tab_play, tab_label = st.tabs(["▶ Play", "✂ Label"])

# ── Play tab ─────────────────────────────────────────────────────────────────
with tab_play:
    st.video(video_path)

# ── Label tab ────────────────────────────────────────────────────────────────
with tab_label:
    # Resolve current frame & keypoints before the columns so both can use them.
    # kp_record is loaded regardless of show_skeleton so the person picker always works.
    frame_idx: int = st.session_state["frame_idx"]
    kp_record: Optional[dict] = _load_keypoints_for_frame(kp_path, frame_idx)
    n_persons: int = len(kp_record["detections"]) if kp_record else 0

    col_vid, col_ctrl = st.columns([3, 1], gap="medium")

    # ── Video column ─────────────────────────────────────────────────────────
    with col_vid:
        # Frame slider – no key, so Streamlit doesn't own its widget state.
        # Buttons update st.session_state["frame_idx"] and rerun; the slider
        # then re-renders at the new value automatically.
        slider_val = st.slider(
            "Frame",
            min_value=0,
            max_value=max(0, total_frames - 1),
            value=frame_idx,
            step=1,
            help="Drag or click to navigate frames",
            label_visibility="collapsed",
        )
        if slider_val != frame_idx:
            st.session_state["frame_idx"] = slider_val
            frame_idx = slider_val
            kp_record = _load_keypoints_for_frame(kp_path, frame_idx)
            n_persons = len(kp_record["detections"]) if kp_record else 0

        # Render frame
        frame_img = _read_frame(video_path, frame_idx)
        if frame_img is not None:
            if show_skeleton and kp_record:
                highlight_idx = st.session_state.get("selected_person_idx")
                frame_img = _overlay_keypoints(frame_img, kp_record, highlight_idx=highlight_idx)
            st.image(frame_img, width='stretch')
        else:
            st.warning(f"Could not read frame {frame_idx}.")

        time_s = frame_idx / fps
        st.caption(
            f"Frame **{frame_idx}** / {total_frames - 1}  •  "
            f"**{_fmt_time(time_s)}**  •  {fps:.1f} fps"
        )

    # ── Controls column ───────────────────────────────────────────────────────
    with col_ctrl:
        # ---- Navigation ----
        st.subheader("Navigation")
        n1, n2, n3, n4 = st.columns(4)
        step_map = {"◀◀": -10, "◀": -1, "▶": 1, "▶▶": 10}
        for _c, (_label, _delta) in zip([n1, n2, n3, n4], step_map.items()):
            with _c:
                if st.button(_label, width='stretch', key=f"nav_{_label}"):
                    new_idx = max(0, min(total_frames - 1, frame_idx + _delta))
                    st.session_state["frame_idx"] = new_idx
                    st.rerun()

        # Jump-to-frame
        jump_frame = st.number_input(
            "Jump to frame",
            min_value=0,
            max_value=total_frames - 1,
            value=frame_idx,
            step=1,
            key="_jump_input",
        )
        if st.button("Go", key="btn_goto", width='stretch'):
            new_idx = int(jump_frame)
            st.session_state["frame_idx"] = new_idx
            st.rerun()

        st.divider()

        # ---- Person selection ----
        st.subheader("Thrower")
        if n_persons > 0:
            cur_person = min(st.session_state.get("selected_person_idx", 0), n_persons - 1)
            chosen = st.radio(
                "Select the person throwing",
                options=[f"Person {i}" for i in range(n_persons)],
                index=cur_person,
                horizontal=True,
                label_visibility="collapsed",
                key="_person_radio",
            )
            st.session_state["selected_person_idx"] = int(chosen.split()[-1])
        else:
            st.caption("No persons detected in this frame.")

        st.divider()

        # ---- Labeling ----
        st.subheader("Label throw")

        pending: Optional[int] = st.session_state.get("pending_start")
        pending_person: Optional[int] = st.session_state.get("pending_person_idx")
        if pending is not None:
            st.info(
                f"**Start:** frame {pending}  \n"
                f"({_fmt_time(pending / fps)})  \n"
                f"**Person:** {pending_person}  \n\n"
                "Navigate to the end of the throw and click **Mark End**."
            )
        else:
            st.caption("Navigate to the first frame of a throw and click **Mark Start**.")

        if st.button("🎯 Mark Start", width='stretch', type="primary"):
            st.session_state["pending_start"] = frame_idx
            st.session_state["pending_person_idx"] = st.session_state.get("selected_person_idx", 0)
            st.rerun()

        end_btn_disabled = pending is None
        if st.button(
            "🏁 Mark End",
            width='stretch',
            disabled=end_btn_disabled,
            type="secondary",
        ):
            start = st.session_state["pending_start"]
            person_idx = st.session_state.get("pending_person_idx", 0)
            end = frame_idx
            if end <= start:
                st.error("End frame must be after start frame.")
            else:
                st.session_state["throws"].append(
                    {
                        "id": len(st.session_state["throws"]),
                        "person_idx": person_idx,
                        "start_frame": start,
                        "end_frame": end,
                        "start_time_s": round(start / fps, 3),
                        "end_time_s": round(end / fps, 3),
                        "duration_s": round((end - start) / fps, 3),
                    }
                )
                st.session_state["pending_start"] = None
                st.session_state["pending_person_idx"] = None
                st.rerun()

        if st.button("↩ Undo last throw", width='stretch'):
            if st.session_state["throws"]:
                st.session_state["throws"].pop()
            st.session_state["pending_start"] = None
            st.rerun()

        st.divider()

        # ---- Persist ----
        if st.button("💾 Save labels", width='stretch', type="primary"):
            payload = {
                "session": selected_name,
                "video": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "throws": st.session_state["throws"],
            }
            labels_path.write_text(json.dumps(payload, indent=2))
            st.success(f"Saved {len(st.session_state['throws'])} throw(s) to labels.json")

# --------------------------------------------------------------------------- #
# Labeled throws table
# --------------------------------------------------------------------------- #
st.divider()
throws: list = st.session_state["throws"]
st.subheader(f"Labeled Throws ({len(throws)})")

if throws:
    df = pd.DataFrame(throws)
    cols = ["id", "person_idx", "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    st.dataframe(df[cols], width='stretch', hide_index=True)

    st.markdown("**Jump to throw:**")
    jump_options = {
        f"Throw {t['id']}  –  {_fmt_time(t['start_time_s'])} → {_fmt_time(t['end_time_s'])}  "
        f"({t['duration_s']:.2f}s)": t
        for t in throws
    }
    jump_sel = st.selectbox(
        "Select throw", ["—"] + list(jump_options.keys()), label_visibility="collapsed"
    )
    if jump_sel != "—":
        target = jump_options[jump_sel]
        jc1, jc2 = st.columns(2)
        with jc1:
            if st.button("▶ Go to Start", width='stretch'):
                st.session_state["frame_idx"] = target["start_frame"]
                st.rerun()
        with jc2:
            if st.button("⏹ Go to End", width='stretch'):
                st.session_state["frame_idx"] = target["end_frame"]
                st.rerun()
else:
    st.info(
        "No throws labeled yet.  \n"
        "Use the **Mark Start** / **Mark End** buttons to annotate each throw."
    )
