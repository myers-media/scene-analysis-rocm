from __future__ import annotations

from pathlib import Path

from PIL import Image

from .pipeline import ScenePipeline, TaskSet
from .types import FrameAnalysis, VideoAnalysis


def _cv2():
    import cv2

    return cv2


def read_rgb_frame(path: str | Path, index: int) -> Image.Image | None:
    import numpy as np

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.asarray(rgb))
    finally:
        capture.release()


def sample_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    max_frames = max(1, min(max_frames, frame_count))
    if max_frames == 1:
        return [0]
    return [int(round(i * (frame_count - 1) / (max_frames - 1))) for i in range(max_frames)]


def analyze_video(
    path: str | Path,
    pipeline: ScenePipeline,
    tasks: TaskSet,
    *,
    max_frames: int = 12,
    conf: float = 0.25,
    narrative=None,
) -> VideoAnalysis:
    import numpy as np

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = sample_frame_indices(frame_count if frame_count > 0 else max_frames, max_frames)
        frames: list[FrameAnalysis] = []
        unique: dict[str, int] = {}
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(np.asarray(rgb))
            analysis = pipeline.analyze_image(
                image, tasks, conf=conf, narrative=narrative, source=f"video:{idx}"
            )
            timestamp = idx / fps if fps else 0.0
            frames.append(FrameAnalysis(frame_index=idx, timestamp_s=round(timestamp, 3), analysis=analysis))
            for label, count in analysis.label_counts().items():
                unique[label] = unique.get(label, 0) + count
        duration = (frame_count / fps) if fps and frame_count else 0.0
        return VideoAnalysis(
            path=str(path),
            fps=round(fps, 3),
            frame_count=frame_count,
            duration_s=round(duration, 3),
            frames=frames,
            unique_labels=unique,
        )
    finally:
        capture.release()
