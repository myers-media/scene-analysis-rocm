from __future__ import annotations

import numpy as np
from PIL import Image

from .types import ColorSwatch, Composition

BLUR_SHARPNESS_THRESHOLD = 40.0


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    return np.asarray(rgb, dtype=np.float32)


def _kmeans_colors(pixels: np.ndarray, k: int = 5, iters: int = 8, seed: int = 7) -> list[ColorSwatch]:
    if pixels.size == 0:
        return []
    n = pixels.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = pixels[rng.choice(n, size=k, replace=False)].astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        distances = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = pixels[mask].mean(axis=0)
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    order = np.argsort(-counts)
    swatches: list[ColorSwatch] = []
    total = float(n)
    for i in order:
        if counts[i] <= 0:
            continue
        r, g, b = np.clip(np.round(centers[i]), 0, 255).astype(int)
        swatches.append(ColorSwatch(r=int(r), g=int(g), b=int(b), fraction=float(counts[i] / total)))
    return swatches


def _laplacian_var(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    kernel_sum = (
        padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(kernel_sum.var())


def _thirds_score(gray: np.ndarray) -> float:
    """How much gradient energy sits near the rule-of-thirds lines (0-1)."""
    h, w = gray.shape
    gy, gx = np.gradient(gray)
    energy = np.hypot(gx, gy)
    total = float(energy.sum()) + 1e-6
    ys = [h / 3.0, 2.0 * h / 3.0]
    xs = [w / 3.0, 2.0 * w / 3.0]
    band_y = max(1, int(round(0.04 * h)))
    band_x = max(1, int(round(0.04 * w)))
    mask = np.zeros_like(energy, dtype=bool)
    for y in ys:
        lo = max(0, int(y) - band_y)
        hi = min(h, int(y) + band_y)
        mask[lo:hi, :] = True
    for x in xs:
        lo = max(0, int(x) - band_x)
        hi = min(w, int(x) + band_x)
        mask[:, lo:hi] = True
    return float(energy[mask].sum() / total)


def analyze_composition(image: Image.Image, color_k: int = 5) -> Composition:
    arr = _as_rgb_array(image)
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    saturation = float(sat.mean())
    sharpness = _laplacian_var(gray)
    thirds = _thirds_score(gray)

    step = max(1, int(np.sqrt((h * w) / 8000)))
    sample = arr[::step, ::step].reshape(-1, 3)
    colors = _kmeans_colors(sample, k=color_k)

    return Composition(
        width=int(w),
        height=int(h),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        saturation=round(saturation, 4),
        sharpness=round(sharpness, 2),
        is_blurry=sharpness < BLUR_SHARPNESS_THRESHOLD,
        rule_of_thirds_score=round(thirds, 4),
        dominant_colors=colors,
    )
