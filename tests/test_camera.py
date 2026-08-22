from __future__ import annotations

from scene_analysis.camera import live_taskset, parse_camera_source
from scene_analysis.pipeline import TaskSet


def test_parse_camera_index():
    assert parse_camera_source("0") == 0
    assert parse_camera_source(2) == 2
    assert parse_camera_source(" 3 ") == 3


def test_parse_camera_url():
    assert parse_camera_source("rtsp://10.0.0.8/live") == "rtsp://10.0.0.8/live"
    assert parse_camera_source("/dev/video0") == "/dev/video0"


def test_live_taskset_detects_every_frame_and_tags_periodically():
    base = TaskSet(detect=True, scene=True, composition=True, caption=True, depth=True, grok=True)
    first = live_taskset(base, 0, 15)
    assert first.detect and first.scene and first.composition
    assert not first.caption and not first.depth and not first.grok
    mid = live_taskset(base, 3, 15)
    assert mid.detect and not mid.scene and not mid.composition
    refresh = live_taskset(base, 15, 15)
    assert refresh.scene and refresh.composition


def test_live_taskset_can_enable_slow_models():
    base = TaskSet(detect=True, scene=False, caption=True, depth=True)
    tasks = live_taskset(base, 0, 10, allow_slow=True)
    assert tasks.caption and tasks.depth
    assert not tasks.grok
