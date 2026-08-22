from __future__ import annotations

from types import SimpleNamespace

from scene_analysis.models.scene_clip import embedding_from_clip_output


class _FakeTensor:
    def __init__(self, name: str):
        self.name = name

    def norm(self, dim=-1, keepdim=True):
        return self


def test_unwrap_transformers_v5_pooling_output():
    pooled = _FakeTensor("pooled")
    output = SimpleNamespace(pooler_output=pooled, last_hidden_state=_FakeTensor("hidden"))
    assert embedding_from_clip_output(output) is pooled


def test_unwrap_transformers_v4_tensor():
    tensor = _FakeTensor("raw")
    assert embedding_from_clip_output(tensor) is tensor


def test_unwrap_tuple_output():
    first = _FakeTensor("first")
    assert embedding_from_clip_output((first, _FakeTensor("second"))) is first
