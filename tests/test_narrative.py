from __future__ import annotations

from PIL import Image

from scene_analysis.llm.lmstudio import (
    DEFAULT_MODEL,
    build_chat_messages,
    ids_from_models_payload,
    match_loaded_model,
    server_origin,
)
from scene_analysis.llm.narrative import NarrativeConfig
from scene_analysis.pipeline import TaskSet
from scene_analysis.types import BoundingBox, SceneAnalysis, SceneTag


def _analysis() -> SceneAnalysis:
    return SceneAnalysis(
        width=10,
        height=10,
        caption="a red car on a street",
        detections=[BoundingBox(0, 0, 5, 5, "car", 0.9)],
        scene_tags=[SceneTag("city street", 0.4)],
    )


def test_default_provider_is_lmstudio():
    cfg = NarrativeConfig()
    assert cfg.normalized_provider() == "lmstudio"
    assert DEFAULT_MODEL == "qwen/qwen3.8-27b"


def test_provider_aliases():
    assert NarrativeConfig(provider="LM Studio").normalized_provider() == "lmstudio"
    assert NarrativeConfig(provider="xai").normalized_provider() == "grok"
    assert NarrativeConfig(provider="grok-4.6").normalized_provider() == "grok"


def test_unknown_provider_raises():
    try:
        NarrativeConfig(provider="openai").normalized_provider()
    except ValueError as exc:
        assert "openai" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_narrative_task_alias():
    assert TaskSet.from_names(["narrative"]).grok
    assert not TaskSet.from_names(["detect"]).grok


def test_server_origin_strips_v1():
    assert server_origin("http://localhost:1234/v1") == "http://localhost:1234"
    assert server_origin("http://127.0.0.1:1234") == "http://127.0.0.1:1234"


def test_ids_from_v0_and_v1_payloads_skip_embeddings():
    v0 = {
        "data": [
            {"id": "qwen/qwen3.8-27b", "type": "llm", "state": "loaded"},
            {"id": "mistral-7b", "type": "llm", "state": "not-loaded"},
            {"id": "nomic-embed", "type": "embedding"},
        ]
    }
    v1 = {
        "models": [
            {"key": "google/gemma-3-12b", "type": "llm"},
            {"modelKey": "text-embed-foo", "type": "embeddings"},
        ]
    }
    assert ids_from_models_payload(v0) == ["qwen/qwen3.8-27b", "mistral-7b"]
    assert ids_from_models_payload(v1) == ["google/gemma-3-12b"]


def test_match_loaded_model_uses_exact_or_single_loaded():
    loaded = ["Qwen3.8-27B-Instruct-GGUF"]
    assert match_loaded_model("qwen/qwen3.8-27b", loaded) == loaded[0]
    assert match_loaded_model("Qwen3.8-27B-Instruct-GGUF", loaded) == loaded[0]


def test_match_loaded_model_keeps_request_when_ambiguous():
    loaded = ["alpha-7b", "beta-13b"]
    assert match_loaded_model("qwen/qwen3.8-27b", loaded) == "qwen/qwen3.8-27b"


def test_lmstudio_messages_with_and_without_image():
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    analysis = _analysis()
    text_only = build_chat_messages(image, analysis, include_image=False)
    assert text_only[0]["role"] == "user"
    assert "car" in text_only[0]["content"]
    with_image = build_chat_messages(image, analysis, include_image=True)
    parts = with_image[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
