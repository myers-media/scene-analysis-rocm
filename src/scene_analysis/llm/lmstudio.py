from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from ..types import SceneAnalysis
from .common import image_data_url, narrative_prompt

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


def resolve_lmstudio_url(base_url: str | None = None) -> str:
    url = (base_url or os.environ.get("LM_STUDIO_BASE_URL") or DEFAULT_BASE_URL).strip()
    return url.rstrip("/")


def resolve_lmstudio_model(model: str | None = None) -> str:
    return (model or os.environ.get("LM_STUDIO_MODEL") or DEFAULT_MODEL).strip()


def _normalize_id(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def match_loaded_model(requested: str, loaded: list[str]) -> str:
    """Map a configured id onto whatever LM Studio actually has loaded.

    Sending a Hugging Face-style id that is not loaded makes some LM Studio
    builds try to download it and fail with 'Engine protocol ... fetch failed'.
    """
    if not loaded:
        return requested
    wanted = _normalize_id(requested)
    wanted_tail = wanted.split("/")[-1]
    for name in loaded:
        if _normalize_id(name) == wanted:
            return name
    for name in loaded:
        tail = _normalize_id(name).split("/")[-1]
        if tail == wanted_tail or wanted_tail in tail or tail in wanted_tail:
            return name
    if len(loaded) == 1:
        return loaded[0]
    return requested


def _client(base_url: str, api_key: str | None):
    from openai import OpenAI

    key = api_key or os.environ.get("LM_STUDIO_API_KEY") or DEFAULT_API_KEY
    return OpenAI(api_key=key, base_url=base_url, timeout=180.0)


def server_origin(base_url: str | None = None) -> str:
    url = resolve_lmstudio_url(base_url)
    if url.endswith("/v1"):
        return url[: -len("/v1")]
    return url


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = api_key or os.environ.get("LM_STUDIO_API_KEY") or DEFAULT_API_KEY
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def ids_from_models_payload(payload: Any) -> list[str]:
    """Parse LM Studio v0/v1 and OpenAI-style model lists. Skip embedding-only rows."""
    if isinstance(payload, dict):
        items = payload.get("models") or payload.get("data") or payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    loaded: list[str] = []
    rest: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = None
        kind = ""
        state = ""
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            kind = str(item.get("type") or item.get("modelType") or item.get("model_type") or "").lower()
            state = str(item.get("state") or item.get("status") or "").lower()
            for key in ("id", "key", "modelKey", "model_key", "name"):
                if item.get(key):
                    name = str(item.get(key))
                    break
        if not name or name in seen:
            continue
        if kind in {"embedding", "embeddings"}:
            continue
        seen.add(name)
        if state in {"loaded", "active"}:
            loaded.append(name)
        else:
            rest.append(name)
    return loaded + sorted(rest, key=str.lower)


def _http_json(url: str, api_key: str | None, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, headers=_auth_headers(api_key), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_via_lms_cli() -> list[str]:
    try:
        proc = subprocess.run(
            ["lms", "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return ids_from_models_payload(payload)


def list_lmstudio_models(base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """All downloaded/available LLMs, not only the one currently loaded."""
    origin = server_origin(base_url)
    errors: list[str] = []
    for path in ("/api/v1/models", "/api/v0/models"):
        try:
            names = ids_from_models_payload(_http_json(origin + path, api_key))
            if names:
                return names
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"{path}: {exc}")
    try:
        client = _client(resolve_lmstudio_url(base_url), api_key)
        listing = client.models.list()
        names = ids_from_models_payload(
            {"data": [{"id": getattr(item, "id", None)} for item in (getattr(listing, "data", None) or [])]}
        )
        if names:
            return names
    except Exception as exc:
        errors.append(f"/v1/models: {exc}")
    cli_names = _list_via_lms_cli()
    if cli_names:
        return cli_names
    if errors:
        raise RuntimeError("; ".join(errors))
    return []


def build_chat_messages(
    image: Image.Image,
    analysis: SceneAnalysis,
    *,
    include_image: bool,
) -> list[dict]:
    prompt = narrative_prompt(analysis)
    if not include_image:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            ],
        }
    ]


def _completion_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return str(response)
    message = choices[0].message
    content = getattr(message, "content", None)
    if content:
        return str(content).strip()
    return str(response)


def enrich_with_lmstudio(
    image: Image.Image,
    analysis: SceneAnalysis,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    include_image: bool = False,
) -> str:
    """Ask a model served by LM Studio (OpenAI-compatible local server)."""
    url = resolve_lmstudio_url(base_url)
    requested = resolve_lmstudio_model(model)
    client = _client(url, api_key)
    try:
        loaded = list_lmstudio_models(url, api_key)
    except Exception as exc:
        raise RuntimeError(
            f"Could not list models at {url}. Start LM Studio's local server. ({exc})"
        ) from exc
    if not loaded:
        raise RuntimeError(
            f"LM Studio at {url} has no loaded model. Load {requested} (or any chat model) and try again."
        )
    model_id = match_loaded_model(requested, loaded)

    def _request(with_image: bool) -> str:
        response = client.chat.completions.create(
            model=model_id,
            messages=build_chat_messages(image, analysis, include_image=with_image),
            temperature=0.3,
            max_tokens=400,
        )
        return _completion_text(response)

    if include_image:
        try:
            return _request(True)
        except Exception:
            time.sleep(0.4)

    try:
        return _request(False)
    except Exception as exc:
        raise RuntimeError(
            f"LM Studio request failed ({url} model={model_id}; requested {requested!r}; "
            f"loaded={loaded}). Use the exact id from 'List LM Studio models'. "
            "qwen/qwen3.8-27b is text-only — leave 'Send snapshot' unchecked. "
            f"Error: {exc}"
        ) from exc
