from __future__ import annotations

import atexit
import sys
import threading
import time
from dataclasses import dataclass

from PIL import Image

from .pipeline import TaskSet


def parse_camera_source(value: str | int) -> int | str:
    """Accept a device index, RTSP/HTTP URL, or video device path."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def live_taskset(base: TaskSet, frame_index: int, tag_every: int, *, allow_slow: bool = False) -> TaskSet:
    """Per-frame live tasks: detect every frame, refresh tags periodically, skip slow models."""
    periodic = frame_index == 0 or (tag_every > 0 and frame_index % tag_every == 0)
    return TaskSet(
        detect=base.detect,
        scene=base.scene and periodic,
        caption=bool(allow_slow and base.caption and periodic),
        depth=bool(allow_slow and base.depth and periodic),
        composition=base.composition and periodic,
        grok=False,
    )


def _preferred_backends(source: int | str) -> list[int]:
    import cv2

    if not isinstance(source, int):
        return [cv2.CAP_FFMPEG, cv2.CAP_ANY]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def list_camera_indices(max_index: int = 6) -> list[int]:
    """Probe low device indexes. May take a second; skip if you already know the index."""
    import cv2

    found: list[int] = []
    for index in range(max_index):
        opened = False
        for backend in _preferred_backends(index):
            cap = cv2.VideoCapture(index, backend)
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        found.append(index)
                        opened = True
                        break
            finally:
                cap.release()
        if opened:
            continue
    return found


@dataclass
class CameraStream:
    source: int | str
    width: int = 1280
    height: int = 720
    fps_request: float = 30.0

    def __post_init__(self) -> None:
        self.source = parse_camera_source(self.source)
        self._cap = None
        self._backend = None
        self._open()

    def _open(self) -> None:
        import cv2

        last_error = None
        for backend in _preferred_backends(self.source):
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened():
                cap.release()
                last_error = f"backend {backend} failed"
                continue
            if isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                cap.set(cv2.CAP_PROP_FPS, float(self.fps_request))
            ok, frame = False, None
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
            if not ok or frame is None:
                cap.release()
                last_error = "opened but produced no frame"
                continue
            self._cap = cap
            self._backend = backend
            atexit.register(self.release)
            return
        raise RuntimeError(
            f"Could not open camera source {self.source!r}. {last_error or ''} "
            "Check that the device is not in use (Zoom/Teams), try another index, "
            "or pass an RTSP URL."
        )

    @property
    def is_open(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def read_rgb(self) -> Image.Image | None:
        import cv2
        import numpy as np

        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.asarray(rgb))

    def release(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is None:
            return
        # DirectShow/MSMF cap.release() can deadlock on Windows; don't block shutdown.
        worker = threading.Thread(target=cap.release, name="opencv-release", daemon=True)
        worker.start()
        worker.join(timeout=1.5)


def run_live_window(
    pipeline,
    source: int | str,
    tasks: TaskSet,
    *,
    conf: float = 0.25,
    tag_every: int = 15,
    width: int = 1280,
    height: int = 720,
    allow_slow: bool = False,
) -> None:
    """OpenCV preview window. Press Q to quit."""
    import cv2
    import numpy as np

    from .viz import draw_detections

    stream = CameraStream(source, width=width, height=height)
    last_tags = []
    frame_index = 0
    fps = 0.0
    last = time.perf_counter()
    try:
        while True:
            image = stream.read_rgb()
            if image is None:
                break
            frame_tasks = live_taskset(tasks, frame_index, tag_every, allow_slow=allow_slow)
            analysis = pipeline.analyze_image(image, frame_tasks, conf=conf, source="live")
            if analysis.scene_tags:
                last_tags = analysis.scene_tags
            elif last_tags:
                analysis.scene_tags = last_tags
            annotated = draw_detections(image, analysis.detections, analysis.scene_tags)
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                fps = (0.85 * fps) + (0.15 / dt) if fps else 1.0 / dt
            bgr = cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"{pipeline.device.backend.upper()}  {fps:.1f} FPS  q=quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (237, 28, 36),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("ROCm Scene Analysis — live", bgr)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
