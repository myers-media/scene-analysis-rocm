from __future__ import annotations

from scene_analysis.cli import build_parser


def test_parser_probe_flag():
    args = build_parser().parse_args(["--probe"])
    assert args.probe is True
    assert args.input is None


def test_parser_live_flags():
    args = build_parser().parse_args(["--live", "--camera", "rtsp://10.0.0.8/stream", "--tag-every", "10"])
    assert args.live is True
    assert args.camera == "rtsp://10.0.0.8/stream"
    assert args.tag_every == 10


def test_parser_image_defaults():
    args = build_parser().parse_args(["photo.jpg", "--overlay", "out.png"])
    assert args.input == "photo.jpg"
    assert args.overlay == "out.png"
    assert args.tasks == "detect,scene,composition"
    assert args.narrative_provider == "lmstudio"
    assert args.llm_image is False


def test_parser_grok_narrative_provider():
    args = build_parser().parse_args(
        ["photo.jpg", "--tasks", "narrative", "--narrative-provider", "grok", "--llm-model", "grok-4.6"]
    )
    assert args.narrative_provider == "grok"
    assert args.llm_model == "grok-4.6"
