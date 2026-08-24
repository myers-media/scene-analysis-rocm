# Scene Analysis ROCm — Grok notes

Project root: `C:\Users\jeff\scene-analysis-rocm` (never `C:\WINDOWS\system32`).

Full session recap: `docs/SESSION-2026-08-22.md`. User-facing docs: `README.md` — update it when behavior changes.

## Product rules

- AMD ROCm via PyTorch HIP (`torch.cuda` API when `torch.version.hip` is set). CPU/DirectML fallback on Windows.
- Streamlit UI in `app.py`; shared library in `src/scene_analysis`.
- Default LLM narrative: **LM Studio** `qwen/qwen3.8-27b` at `http://localhost:1234/v1`. Keep **Grok 4.6** as an option (`XAI_API_KEY`).
- Live camera: **This browser** uses the viewer’s webcam (same picker as Camera snapshot). **Streamlit host** uses OpenCV on the server. Do **not** write live frames to disk. Uploaded videos may go under `outputs/`.
- Remote **Camera snapshot** needs HTTPS. `scripts/run_ui.py` auto-generates `certs/cert.pem` + `certs/key.pem` and starts Streamlit with SSL on `0.0.0.0:8501`. Do not commit private keys.
- `run.sh` must use `.venv/bin/python` **or** `.venv/Scripts/python.exe` (Windows venvs have no `bin/activate`). Prefer `run.ps1` in PowerShell.
- CLIP must unwrap Transformers 5 `BaseModelOutputWithPooling` via `pooler_output`.
- LM Studio: text-only by default (no image). Catalog dropdown from `/api/v1/models`, `/api/v0/models`, then `/v1/models`, then `lms ls --json`. Map configured ids to loaded/downloaded names to avoid `fetch failed`.

## Tests

`pytest` from repo root (`pythonpath = src`). Do not require GPU weights, a camera, or a live LLM.
