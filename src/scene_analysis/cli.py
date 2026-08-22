from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .device import probe_rocm
from .pipeline import ScenePipeline, TaskSet
from .video import analyze_video
from .viz import draw_detections, png_bytes


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a scene with AMD ROCm-backed vision models.")
    parser.add_argument("input", nargs="?", help="Image or video path")
    parser.add_argument("--tasks", default="detect,scene,composition", help="Comma-separated tasks")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--json", dest="json_out", help="Write analysis JSON to this path")
    parser.add_argument("--overlay", help="Write annotated PNG to this path")
    parser.add_argument("--keep-loaded", action="store_true", help="Keep models resident in VRAM")
    parser.add_argument("--probe", action="store_true", help="Print ROCm/PyTorch diagnostics and exit")
    parser.add_argument("--max-frames", type=int, default=8, help="Video frames to sample")
    parser.add_argument("--live", action="store_true", help="Open a live camera / RTSP preview window")
    parser.add_argument("--camera", default="0", help="Camera index, device path, or RTSP/HTTP URL")
    parser.add_argument("--tag-every", type=int, default=15, help="Refresh CLIP/composition every N live frames")
    parser.add_argument("--width", type=int, default=1280, help="Capture width for live camera")
    parser.add_argument("--height", type=int, default=720, help="Capture height for live camera")
    parser.add_argument("--allow-slow", action="store_true", help="Allow caption/depth during live capture")
    parser.add_argument(
        "--narrative-provider",
        default="lmstudio",
        choices=["lmstudio", "grok"],
        help="LLM for scene narrative: local LM Studio or Grok 4.6",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Narrative model id (default qwen/qwen3.8-27b for LM Studio, grok-4.6 for Grok)",
    )
    parser.add_argument(
        "--lm-studio-url",
        default=None,
        help="LM Studio OpenAI-compatible base URL (default http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--llm-image",
        action="store_true",
        help="Send the photo to LM Studio (vision checkpoints only; off by default)",
    )
    parser.add_argument(
        "--no-llm-image",
        action="store_true",
        help="Deprecated: photos are omitted unless --llm-image is set",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.probe:
        print(json.dumps(probe_rocm(), indent=2))
        return 0
    if args.live:
        from .camera import run_live_window

        tasks = TaskSet.from_names(args.tasks.split(","))
        pipeline = ScenePipeline(keep_models_loaded=True)
        run_live_window(
            pipeline,
            args.camera,
            tasks,
            conf=args.conf,
            tag_every=args.tag_every,
            width=args.width,
            height=args.height,
            allow_slow=args.allow_slow,
        )
        return 0
    if not args.input:
        parser.error("input path is required unless --probe or --live is set")

    path = Path(args.input)
    if not path.exists():
        parser.error(f"file not found: {path}")

    tasks = TaskSet.from_names(args.tasks.split(","))
    pipeline = ScenePipeline(keep_models_loaded=args.keep_loaded)
    from .llm.narrative import NarrativeConfig

    narrative = NarrativeConfig(
        provider=args.narrative_provider,
        model=args.llm_model,
        base_url=args.lm_studio_url,
        include_image=bool(args.llm_image) and not args.no_llm_image,
    )
    suffix = path.suffix.lower()

    if suffix in VIDEO_SUFFIXES:
        video = analyze_video(
            path,
            pipeline,
            tasks,
            max_frames=args.max_frames,
            conf=args.conf,
            narrative=narrative,
        )
        payload = {
            "path": video.path,
            "fps": video.fps,
            "frame_count": video.frame_count,
            "duration_s": video.duration_s,
            "unique_labels": video.unique_labels,
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "timestamp_s": f.timestamp_s,
                    "analysis": f.analysis.to_dict(),
                }
                for f in video.frames
            ],
        }
        text = json.dumps(payload, indent=2)
        if args.json_out:
            Path(args.json_out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0

    if suffix not in IMAGE_SUFFIXES:
        parser.error(f"unsupported file type: {suffix}")

    image = Image.open(path)
    analysis = pipeline.analyze_image(
        image, tasks, conf=args.conf, narrative=narrative, source=str(path)
    )
    if args.overlay:
        annotated = draw_detections(image.convert("RGB"), analysis.detections, analysis.scene_tags)
        Path(args.overlay).write_bytes(png_bytes(annotated))
    text = json.dumps(analysis.to_dict(), indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    else:
        print(text)
    if analysis.warnings:
        for warning in analysis.warnings:
            print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
