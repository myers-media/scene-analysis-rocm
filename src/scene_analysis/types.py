from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class SceneTag:
    label: str
    score: float


@dataclass
class ColorSwatch:
    r: int
    g: int
    b: int
    fraction: float

    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


@dataclass
class Composition:
    width: int
    height: int
    brightness: float
    contrast: float
    saturation: float
    sharpness: float
    is_blurry: bool
    rule_of_thirds_score: float
    dominant_colors: list[ColorSwatch] = field(default_factory=list)


@dataclass
class DepthMap:
    width: int
    height: int
    min_value: float
    max_value: float
    mean_value: float
    preview_png: bytes


@dataclass
class Timing:
    task: str
    seconds: float
    backend: str


@dataclass
class DeviceSnapshot:
    backend: str
    name: str
    index: int
    memory_total_gb: float | None
    hip_version: str | None
    pytorch_version: str | None
    rocm_system: str | None


@dataclass
class FrameAnalysis:
    frame_index: int
    timestamp_s: float
    analysis: SceneAnalysis


@dataclass
class VideoAnalysis:
    path: str
    fps: float
    frame_count: int
    duration_s: float
    frames: list[FrameAnalysis]
    unique_labels: dict[str, int]


@dataclass
class SceneAnalysis:
    width: int
    height: int
    caption: str | None = None
    detections: list[BoundingBox] = field(default_factory=list)
    scene_tags: list[SceneTag] = field(default_factory=list)
    composition: Composition | None = None
    depth: DepthMap | None = None
    grok_narrative: str | None = None
    timings: list[Timing] = field(default_factory=list)
    device: DeviceSnapshot | None = None
    warnings: list[str] = field(default_factory=list)
    source: str = "image"

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.depth is not None:
            payload["depth"] = {
                "width": self.depth.width,
                "height": self.depth.height,
                "min_value": self.depth.min_value,
                "max_value": self.depth.max_value,
                "mean_value": self.depth.mean_value,
                "preview_png_bytes": len(self.depth.preview_png),
            }
        return payload
