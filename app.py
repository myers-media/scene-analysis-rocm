from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

import streamlit as st

from scene_analysis.camera import CameraStream, list_camera_indices, live_taskset
from scene_analysis.demo import make_demo_image
from scene_analysis.device import detect_device, gpu_memory_allocated_gb, matmul_gflops, probe_rocm
from scene_analysis.llm.lmstudio import DEFAULT_BASE_URL as LM_STUDIO_URL
from scene_analysis.llm.lmstudio import DEFAULT_MODEL as LM_STUDIO_MODEL
from scene_analysis.llm.lmstudio import list_lmstudio_models, match_loaded_model
from scene_analysis.llm.narrative import NarrativeConfig
from scene_analysis.pipeline import ScenePipeline, TaskSet
from scene_analysis.video import analyze_video, read_rgb_frame
from scene_analysis.viz import draw_detections
from scene_analysis.web_camera import capture_browser_camera

st.set_page_config(
    page_title="ROCm Scene Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 1.2rem; }
.hero {
  background: linear-gradient(120deg, #1a0a0c 0%, #191C24 45%, #0f2a33 100%);
  border: 1px solid #2a303c;
  border-radius: 18px;
  padding: 1.4rem 1.6rem 1.2rem 1.6rem;
  margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 1.8rem; letter-spacing: -0.02em; }
.hero p { margin: 0.35rem 0 0 0; color: #c7c2ba; }
.badge {
  display: inline-block;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-right: 0.4rem;
  margin-top: 0.7rem;
}
.badge-rocm { background: #ED1C24; color: white; }
.badge-cuda { background: #76b900; color: #0b1200; }
.badge-cpu { background: #3b4252; color: #d8dee9; }
.badge-dml { background: #0078d4; color: white; }
.badge-off { background: #4c3228; color: #ffd0c0; }
.swatch {
  width: 100%;
  height: 28px;
  border-radius: 6px;
  border: 1px solid #333;
}
.metric-card {
  background: #191C24;
  border: 1px solid #2a303c;
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
}
.stButton>button {
  background: #ED1C24;
  color: white;
  border: 0;
  font-weight: 600;
}
.stButton>button:hover { background: #c3141c; color: white; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _badge_class(backend: str) -> str:
    return {
        "rocm": "badge-rocm",
        "cuda": "badge-cuda",
        "directml": "badge-dml",
        "cpu": "badge-cpu",
        "unavailable": "badge-off",
    }.get(backend, "badge-cpu")


@st.cache_resource(show_spinner=False)
def load_pipeline(keep_loaded: bool, detector_name: str) -> ScenePipeline:
    return ScenePipeline(keep_models_loaded=keep_loaded, detector_name=detector_name)


@st.cache_resource(show_spinner=False, on_release=lambda cam: cam.release())
def get_live_camera(source: str, width: int, height: int) -> CameraStream:
    return CameraStream(source, width=width, height=height)


def release_live_camera() -> None:
    get_live_camera.clear()


@st.cache_data(show_spinner=False)
def cached_probe() -> dict:
    return probe_rocm()


@st.cache_data(ttl=20, show_spinner=False)
def cached_lmstudio_models(url: str) -> list[str]:
    return list_lmstudio_models(url)


def render_hero(device) -> None:
    mem = f" · {device.memory_total_gb} GB" if device.memory_total_gb else ""
    hip = f" · HIP {device.hip_version}" if device.hip_version else ""
    st.markdown(
        f"""
        <div class="hero">
          <h1>ROCm Scene Analysis</h1>
          <p>Object detection, scene tagging, captions, and depth — accelerated on AMD GPUs through PyTorch HIP.</p>
          <span class="badge {_badge_class(device.backend)}">{device.backend.upper()}</span>
          <span class="badge badge-cpu">{device.name}{mem}{hip}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_swatches(colors) -> None:
    if not colors:
        return
    cols = st.columns(len(colors))
    for col, swatch in zip(cols, colors):
        with col:
            st.markdown(
                f'<div class="swatch" style="background:{swatch.hex}"></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{swatch.hex} · {swatch.fraction:.0%}")


def analysis_overview(image: Image.Image, analysis) -> None:
    annotated = draw_detections(image, analysis.detections, analysis.scene_tags)
    left, right = st.columns((1.35, 1), gap="large")
    with left:
        st.image(annotated, caption="Annotated scene", use_container_width=True)
    with right:
        if analysis.caption:
            st.subheader("Caption")
            st.write(analysis.caption)
        if analysis.scene_tags:
            st.subheader("Scene tags")
            for tag in analysis.scene_tags:
                st.progress(min(1.0, tag.score), text=f"{tag.label}  {tag.score:.0%}")
        counts = analysis.label_counts()
        if counts:
            st.subheader("Objects")
            st.dataframe(
                [{"class": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
                hide_index=True,
                use_container_width=True,
            )
        if analysis.grok_narrative:
            st.subheader("Scene narrative")
            st.write(analysis.grok_narrative)
        if analysis.warnings:
            st.warning("\n".join(analysis.warnings))


def analysis_tabs(image: Image.Image, analysis) -> None:
    tab_names = ["Overview", "Detections", "Composition", "Depth", "Performance", "JSON"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        analysis_overview(image, analysis)
    with tabs[1]:
        if not analysis.detections:
            st.info("No detections. Enable object detection or lower the confidence threshold.")
        else:
            st.dataframe(
                [
                    {
                        "label": d.label,
                        "confidence": round(d.confidence, 3),
                        "x1": round(d.x1, 1),
                        "y1": round(d.y1, 1),
                        "x2": round(d.x2, 1),
                        "y2": round(d.y2, 1),
                    }
                    for d in analysis.detections
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.image(draw_detections(image, analysis.detections), use_container_width=True)
    with tabs[2]:
        if not analysis.composition:
            st.info("Composition was not computed.")
        else:
            c = analysis.composition
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Brightness", f"{c.brightness:.2f}")
            m2.metric("Contrast", f"{c.contrast:.2f}")
            m3.metric("Saturation", f"{c.saturation:.2f}")
            m4.metric("Sharpness", f"{c.sharpness:.0f}", "blurry" if c.is_blurry else "ok")
            st.caption(f"Rule-of-thirds energy: {c.rule_of_thirds_score:.2f}")
            render_swatches(c.dominant_colors)
    with tabs[3]:
        if analysis.depth is None:
            st.info("Enable depth estimation to generate a relative depth map.")
        else:
            st.image(analysis.depth.preview_png, caption="Relative depth (Turbo colormap)", use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("Near", f"{analysis.depth.min_value:.3f}")
            d2.metric("Mean", f"{analysis.depth.mean_value:.3f}")
            d3.metric("Far", f"{analysis.depth.max_value:.3f}")
    with tabs[4]:
        if analysis.timings:
            st.dataframe(
                [{"task": t.task, "seconds": t.seconds, "backend": t.backend} for t in analysis.timings],
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart({t.task: t.seconds for t in analysis.timings})
        allocated = gpu_memory_allocated_gb(detect_device())
        if allocated is not None:
            st.caption(f"Allocated GPU memory after this run: {allocated} GB")
        if analysis.device:
            st.json(
                {
                    "backend": analysis.device.backend,
                    "name": analysis.device.name,
                    "hip_version": analysis.device.hip_version,
                    "pytorch_version": analysis.device.pytorch_version,
                    "rocm_system": analysis.device.rocm_system,
                }
            )
    with tabs[5]:
        payload = json.dumps(analysis.to_dict(), indent=2)
        st.code(payload, language="json")
        st.download_button("Download JSON", payload, file_name="scene_analysis.json", mime="application/json")


def read_image(upload) -> Image.Image:
    return Image.open(upload).convert("RGB")


def save_temp_video(upload) -> Path:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / upload.name
    dest.write_bytes(upload.getvalue())
    return dest


def _init_live_state() -> None:
    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("live_frame_index", 0)
    st.session_state.setdefault("live_last_tags", [])
    st.session_state.setdefault("live_last_counts", {})
    st.session_state.setdefault("live_fps", 0.0)
    st.session_state.setdefault("live_misses", 0)
    st.session_state.setdefault("live_last_tick", None)
    st.session_state.setdefault("live_scan_token", 0)


def _reset_live_counters() -> None:
    st.session_state.live_frame_index = 0
    st.session_state.live_last_tags = []
    st.session_state.live_last_counts = {}
    st.session_state.live_fps = 0.0
    st.session_state.live_misses = 0
    st.session_state.live_last_tick = None


def _show_live_analysis(
    image: Image.Image,
    pipeline: ScenePipeline,
    tasks: TaskSet,
    conf: float,
    tag_every: int,
    allow_slow: bool,
) -> None:
    import time

    frame_index = int(st.session_state.live_frame_index)
    frame_tasks = live_taskset(tasks, frame_index, int(tag_every), allow_slow=allow_slow)
    analysis = pipeline.analyze_image(image, frame_tasks, conf=conf, source="live")
    if analysis.scene_tags:
        st.session_state.live_last_tags = analysis.scene_tags
    else:
        analysis.scene_tags = list(st.session_state.live_last_tags)
    now = time.perf_counter()
    prev = st.session_state.live_last_tick
    if prev:
        dt = now - prev
        if dt > 0:
            prev_fps = float(st.session_state.live_fps)
            inst = 1.0 / dt
            st.session_state.live_fps = (0.8 * prev_fps + 0.2 * inst) if prev_fps else inst
    st.session_state.live_last_tick = now
    st.session_state.live_frame_index = frame_index + 1
    counts = analysis.label_counts()
    if counts:
        st.session_state.live_last_counts = counts
    annotated = draw_detections(image, analysis.detections, analysis.scene_tags)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FPS", f"{st.session_state.live_fps:.1f}")
    m2.metric("Frame", f"{st.session_state.live_frame_index}")
    m3.metric("Objects", f"{len(analysis.detections)}")
    m4.metric("Backend", pipeline.device.backend.upper())
    st.image(annotated, caption="Live annotated feed", use_container_width=True)
    if analysis.warnings:
        st.caption(" · ".join(analysis.warnings[:3]))
    if st.session_state.live_last_counts:
        st.bar_chart(st.session_state.live_last_counts)


def render_live_panel(pipeline: ScenePipeline, tasks: TaskSet, conf: float) -> None:
    """Live analysis from this browser's webcam or from the Streamlit host."""
    _init_live_state()
    origin = st.radio(
        "Live camera location",
        ["This browser (this PC or phone)", "Streamlit host (server)"],
        horizontal=True,
        disabled=st.session_state.live_running,
        key="live_origin",
        help="This browser uses the same device list as Camera snapshot. Host uses OpenCV on the server.",
    )
    browser = origin.startswith("This browser")

    if browser:
        st.info(
            "Frames come from **this device’s webcam** (the computer viewing the page). "
            "Allow access, pick a camera, then Start live feed. "
            "The Streamlit server camera is the other option above."
        )
    else:
        st.info(
            "Frames come from a webcam or RTSP URL on the **Streamlit host**. "
            "This is not the camera on a remote laptop — switch to **This browser** for that."
        )

    left, right = st.columns((2, 1), gap="large")
    source = "0"
    width, height = 1280, 720
    browser_frame = None
    with right:
        resolution = st.selectbox(
            "Capture size (host only)",
            ["640x480", "1280x720", "1920x1080"],
            index=1,
            disabled=st.session_state.live_running or browser,
        )
        width, height = (int(part) for part in resolution.split("x"))
    with left:
        if browser:
            if st.button("Probe local cameras", disabled=st.session_state.live_running):
                st.session_state.live_scan_token = int(st.session_state.live_scan_token) + 1
                st.rerun()
            browser_frame = capture_browser_camera(
                key="live_browser_camera_frames",
                live=st.session_state.live_running,
                interval_ms=300,
                scan_token=int(st.session_state.live_scan_token),
            )
        else:
            host_index = st.selectbox(
                "Host camera index",
                ["0", "1", "2", "3", "4", "5"],
                index=0,
                disabled=st.session_state.live_running,
                help="OpenCV index on the Streamlit server.",
            )
            custom_source = st.text_input(
                "Or RTSP / device path",
                value="",
                disabled=st.session_state.live_running,
            )
            source = custom_source.strip() or host_index

    c1, c2, c3 = st.columns(3)
    tag_every = c1.number_input("Refresh tags every N frames", min_value=1, max_value=120, value=15, step=1)
    allow_slow = c2.checkbox("Allow caption/depth on live", value=False)
    if not browser:
        if c3.button("Probe host cameras", disabled=st.session_state.live_running):
            with st.spinner("Opening camera indexes 0–5 on the server…"):
                found = list_camera_indices(6)
            if found:
                st.success("Host cameras that returned a frame: " + ", ".join(str(i) for i in found))
            else:
                st.warning("No host cameras responded. Try another index or an RTSP URL.")

    b1, b2 = st.columns(2)
    if b1.button("Start live feed", type="primary", disabled=st.session_state.live_running, use_container_width=True):
        st.session_state.live_running = True
        _reset_live_counters()
        if browser:
            release_live_camera()
        st.rerun()
    if b2.button("Stop", disabled=not st.session_state.live_running, use_container_width=True):
        st.session_state.live_running = False
        release_live_camera()
        st.rerun()

    if browser:
        if not st.session_state.live_running:
            st.caption("Allow the camera, pick a device, then Start live feed to run YOLO on this browser’s stream.")
            return
        if browser_frame is None:
            st.info("Waiting for a frame from this browser’s camera…")
            return
        _show_live_analysis(browser_frame, pipeline, tasks, conf, tag_every, allow_slow)
        return

    if not st.session_state.live_running:
        st.caption("Start the feed to run YOLO on every host frame. CLIP/composition refresh on the interval above.")
        return

    @st.fragment(run_every=timedelta(milliseconds=50))
    def live_tick() -> None:
        if not st.session_state.live_running:
            return
        try:
            camera = get_live_camera(source, width, height)
            image = camera.read_rgb()
        except Exception as exc:
            st.session_state.live_running = False
            release_live_camera()
            st.error(str(exc))
            return
        if image is None:
            st.session_state.live_misses = int(st.session_state.live_misses) + 1
            if st.session_state.live_misses > 20:
                st.session_state.live_running = False
                release_live_camera()
                st.error("Host camera stopped producing frames.")
            return
        st.session_state.live_misses = 0
        _show_live_analysis(image, pipeline, tasks, conf, tag_every, allow_slow)

    live_tick()


def main() -> None:
    device = detect_device()
    render_hero(device)

    with st.sidebar:
        st.header("Device")
        st.write(f"**{device.backend.upper()}** · {device.name}")
        if device.pytorch_version:
            st.caption(f"PyTorch {device.pytorch_version}")
        if device.rocm_system:
            st.caption(f"ROCm {device.rocm_system}")
        for note in device.notes:
            st.caption(note)
        if st.button("Probe ROCm stack"):
            st.json(cached_probe())
        if st.button("GEMM smoke test"):
            with st.spinner("Running matmul on the resolved device…"):
                gflops = matmul_gflops(device)
            if gflops is None:
                st.error("Could not run the GEMM probe on this device.")
            else:
                st.success(f"{gflops} GFLOP/s (2048³ FP32)")

        st.divider()
        st.header("Models")
        detector_name = st.selectbox(
            "Detector",
            ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"],
            index=0,
        )
        keep_loaded = st.toggle("Keep models in VRAM", value=device.backend == "rocm")
        conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)

        st.subheader("Tasks")
        detect = st.checkbox("Object detection (YOLO)", True)
        scene = st.checkbox("Scene tags (CLIP)", True)
        composition = st.checkbox("Composition / palette", True)
        caption = st.checkbox("Caption (BLIP)", False)
        depth = st.checkbox("Depth map", False)
        grok = st.checkbox("LLM scene narrative", False)
        narrative_provider = "lmstudio"
        llm_model = LM_STUDIO_MODEL
        lm_studio_url = os.environ.get("LM_STUDIO_BASE_URL", LM_STUDIO_URL)
        include_llm_image = False
        if grok:
            narrative_provider = st.selectbox(
                "Narrative provider",
                ["lmstudio", "grok"],
                index=0,
                format_func=lambda p: (
                    "LM Studio (local)" if p == "lmstudio" else "Grok 4.6 (SpaceXAI)"
                ),
                key="narrative_provider",
            )
            if narrative_provider == "lmstudio":
                lm_studio_url = st.text_input(
                    "LM Studio URL",
                    value=lm_studio_url,
                    help="OpenAI-compatible endpoint. Start the server in LM Studio first.",
                )
                preferred = os.environ.get("LM_STUDIO_MODEL", LM_STUDIO_MODEL)
                catalog_error = None
                try:
                    catalog = cached_lmstudio_models(lm_studio_url)
                except Exception as exc:
                    catalog = []
                    catalog_error = str(exc)
                refresh = st.button("Refresh model list")
                if refresh:
                    cached_lmstudio_models.clear()
                    st.rerun()
                if catalog:
                    matched = match_loaded_model(preferred, catalog)
                    index = catalog.index(matched) if matched in catalog else 0
                    llm_model = st.selectbox(
                        "LM Studio model",
                        catalog,
                        index=index,
                        help="Every downloaded LLM LM Studio reported, not only the one currently loaded.",
                        key="lm_studio_model",
                    )
                    st.caption(f"{len(catalog)} model(s) available.")
                else:
                    if catalog_error:
                        st.warning(
                            f"Could not list LM Studio models at {lm_studio_url}. "
                            "Start the server and click Refresh model list."
                        )
                        st.caption(catalog_error)
                    else:
                        st.warning("LM Studio listed no LLMs. Download a model in the app, then refresh.")
                    llm_model = st.text_input("LM Studio model", value=preferred, key="lm_studio_model_manual")
                include_llm_image = st.checkbox(
                    "Send snapshot to the local model",
                    value=False,
                    help="Leave off for text models such as qwen/qwen3.8-27b. Vision checkpoints only.",
                )
            else:
                llm_model = "grok-4.6"
                if not os.environ.get("XAI_API_KEY"):
                    st.warning("Set XAI_API_KEY in `.env` to use Grok 4.6.")
        max_frames = st.slider("Video sample frames", 3, 24, 8)

        st.divider()
        st.caption("First run downloads model weights (YOLO, CLIP, optional BLIP/Depth).")

    tasks = TaskSet(
        detect=detect,
        scene=scene,
        caption=caption,
        depth=depth,
        composition=composition,
        grok=grok,
    )
    pipeline = load_pipeline(keep_loaded, detector_name)
    narrative_cfg = NarrativeConfig(
        provider=narrative_provider,
        model=llm_model,
        base_url=lm_studio_url if narrative_provider == "lmstudio" else None,
        api_key=os.environ.get("XAI_API_KEY") if narrative_provider == "grok" else os.environ.get("LM_STUDIO_API_KEY"),
        include_image=include_llm_image,
    )

    source = st.radio(
        "Input",
        ["Upload image", "Upload video", "Live camera", "Camera snapshot", "Synthetic demo"],
        horizontal=True,
    )

    image = None
    video_path = None
    if source == "Upload image":
        upload = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upload:
            image = read_image(upload)
    elif source == "Upload video":
        upload = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"])
        if upload:
            video_path = save_temp_video(upload)
    elif source == "Live camera":
        live_pipeline = load_pipeline(True, detector_name)
        render_live_panel(live_pipeline, tasks, conf)
        return
    elif source == "Camera snapshot":
        st.caption(
            "This uses the **webcam on the computer viewing the page**, not the Streamlit host. "
            "Remote access must be HTTPS (https://MACHINE_NAME:8501). "
            "After Allow, pick the correct camera in the dropdown — Streamlit’s default widget "
            "often asks for facingMode=user / device 0, which many PCs do not expose."
        )
        image = capture_browser_camera()
        if image is not None:
            st.image(image, caption="Captured from this browser", use_container_width=True)
    else:
        image = make_demo_image()
        st.caption("Synthetic demo image — useful for checking the UI without a photo.")

    run = st.button("Analyze scene", type="primary", use_container_width=True)

    if not run:
        if image is not None:
            st.image(image, caption="Ready to analyze", use_container_width=True)
        return

    if image is None and video_path is None:
        st.error("Provide an image, video, camera frame, or use the synthetic demo.")
        return

    if video_path is not None:
        with st.spinner("Sampling video frames on the ROCm/CPU pipeline…"):
            video = analyze_video(
                video_path,
                pipeline,
                tasks,
                max_frames=max_frames,
                conf=conf,
                narrative=narrative_cfg,
            )
        st.success(f"Analyzed {len(video.frames)} / {video.frame_count} frames · {video.duration_s:.1f}s @ {video.fps} fps")
        if video.unique_labels:
            st.subheader("Objects across sampled frames")
            st.bar_chart(video.unique_labels)
        for frame in video.frames:
            with st.expander(f"Frame {frame.frame_index}  ·  t={frame.timestamp_s:.2f}s"):
                frame_image = read_rgb_frame(video_path, frame.frame_index) or make_demo_image()
                analysis_tabs(frame_image, frame.analysis)
        return

    with st.spinner("Running scene analysis…"):
        analysis = pipeline.analyze_image(
            image,
            tasks,
            conf=conf,
            narrative=narrative_cfg,
        )
    analysis_tabs(image, analysis)


if __name__ == "__main__":
    main()
