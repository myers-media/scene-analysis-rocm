from __future__ import annotations

from PIL import Image, ImageDraw

from scene_analysis.composition import analyze_composition
from scene_analysis.demo import make_demo_image


def test_composition_on_solid_color():
    image = Image.new("RGB", (64, 48), (255, 0, 0))
    result = analyze_composition(image, color_k=3)
    assert result.width == 64
    assert result.height == 48
    assert 0.2 < result.brightness < 0.5
    assert result.dominant_colors
    top = result.dominant_colors[0]
    assert top.r > 200
    assert top.g < 40
    assert top.b < 40


def test_composition_marks_blurry_flat_image():
    image = Image.new("RGB", (80, 80), (128, 128, 128))
    result = analyze_composition(image)
    assert result.is_blurry
    assert result.sharpness < 40


def test_composition_demo_image_has_palette():
    result = analyze_composition(make_demo_image(320, 180), color_k=4)
    assert len(result.dominant_colors) >= 2
    assert 0 < result.rule_of_thirds_score < 1
    assert result.contrast > 0


def test_high_contrast_edges_are_sharp():
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 90, 90], fill=(255, 255, 255))
    result = analyze_composition(image)
    assert not result.is_blurry
