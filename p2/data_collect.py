"""
data_collect.py — simple clip recorder for gesture datasets (webcam)

Records short labeled video clips (e.g., handshake / none / wave / highfive / fistbump)
at a target FPS for a fixed duration, saved as MP4 + a JSON metadata file.

Usage:
  python data_collect.py --subject S01 --env indoor_bright --out ./dataset

Keys:
  [1] handshake
  [2] none/background
  [3] wave
  [4] highfive
  [5] fistbump
  [r] start recording (uses the last selected label)
  [q] quit

Tips:
- Use 2–3 seconds per clip.
- Collect multiple conditions: distance, lighting, clutter, height mismatch, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass
class ClipMeta:
    clip_id: str
    created_at: str
    subject: str
    env: str
    label: str
    camera_index: int
    fps_target: int
    duration_s: float
    frame_size: Tuple[int, int]
    notes: str


LABEL_KEYS = {
    ord("1"): "handshake",
    ord("2"): "none",
    ord("3"): "wave",
    ord("4"): "highfive",
    ord("5"): "fistbump",
}


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_clip_id(subject: str, label: str) -> str:
    # Example: S01_handshake_20260217_153012
    return f"{subject}_{label}_{now_str()}"


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Request resolution (not guaranteed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def get_frame_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (w, h)


def record_clip(
    cap: cv2.VideoCapture,
    out_video_path: Path,
    fps_target: int,
    duration_s: float,
    overlay_text: str,
) -> Tuple[int, Tuple[int, int]]:
    frame_w, frame_h = get_frame_size(cap)

    # MP4 writer (H.264 availability varies; mp4v is widely supported)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_target, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer. Try a different codec/container.")

    total_frames = int(round(fps_target * duration_s))
    frame_interval = 1.0 / fps_target

    frames_written = 0
    start = time.time()
    next_frame_time = start

    while frames_written < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # Overlay label/countdown for your reference
        remaining = max(0.0, duration_s - (time.time() - start))
        cv2.putText(
            frame,
            f"{overlay_text} | {remaining:0.1f}s left",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frames_written += 1

        # Pace to target FPS (best-effort)
        next_frame_time += frame_interval
        sleep_s = next_frame_time - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

        cv2.imshow("Recorder (press q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    writer.release()
    return frames_written, (frame_w, frame_h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./dataset", help="Output dataset directory")
    ap.add_argument("--subject", type=str, required=True, help="Subject ID, e.g., S01")
    ap.add_argument("--env", type=str, default="unknown", help="Environment tag, e.g., indoor_bright")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument("--duration", type=float, default=2.5, help="Clip duration in seconds")
    ap.add_argument("--width", type=int, default=1280, help="Requested capture width")
    ap.add_argument("--height", type=int, default=720, help="Requested capture height")
    ap.add_argument("--notes", type=str, default="", help="Optional notes stored in metadata")
    args = ap.parse_args()

    out_root = Path(args.out)
    safe_mkdir(out_root)

    cap = open_camera(args.camera, args.width, args.height)
    label = "none"  # default selection

    print("\n=== Controls ===")
    print("  1 handshake | 2 none | 3 wave | 4 highfive | 5 fistbump")
    print("  r record clip with selected label")
    print("  q quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        cv2.putText(
            frame,
            f"Selected label: {label} | Press [r] to record",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Recorder (press q to quit)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in LABEL_KEYS:
            label = LABEL_KEYS[k]
            print(f"Selected: {label}")
        if k == ord("r"):
            clip_id = make_clip_id(args.subject, label)

            # Save into per-label folders (cleaner later)
            out_dir = out_root / label
            safe_mkdir(out_dir)

            video_path = out_dir / f"{clip_id}.mp4"
            meta_path = out_dir / f"{clip_id}.json"

            overlay = f"REC | {args.subject} | {args.env} | {label} | {clip_id}"
            print(f"Recording: {video_path.name}")

            frames_written, frame_size = record_clip(
                cap=cap,
                out_video_path=video_path,
                fps_target=args.fps,
                duration_s=args.duration,
                overlay_text=overlay,
            )

            meta = ClipMeta(
                clip_id=clip_id,
                created_at=datetime.now().isoformat(timespec="seconds"),
                subject=args.subject,
                env=args.env,
                label=label,
                camera_index=args.camera,
                fps_target=args.fps,
                duration_s=args.duration,
                frame_size=frame_size,
                notes=args.notes,
            )

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(asdict(meta), f, indent=2)

            print(f"Saved: {video_path} ({frames_written} frames), meta: {meta_path}\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()