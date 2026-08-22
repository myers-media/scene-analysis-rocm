from __future__ import annotations

from scene_analysis.video import sample_frame_indices


def test_sample_frame_indices_endpoints():
    idx = sample_frame_indices(100, 5)
    assert idx[0] == 0
    assert idx[-1] == 99
    assert len(idx) == 5
    assert idx == sorted(idx)


def test_sample_frame_indices_single():
    assert sample_frame_indices(50, 1) == [0]


def test_sample_empty():
    assert sample_frame_indices(0, 8) == []
