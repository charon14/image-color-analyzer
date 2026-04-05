"""
图片色彩分析工具
支持命令行批量分析，也支持本地网页上传分析。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── 配置 ──────────────────────────────────────────────
IMAGE_DIR = "./images"
MAX_IMAGES = 6
SCATTER_ALPHA = 0.08
SCATTER_SIZE = 1.2
SAMPLE_RATIO = 0.15
MAX_DIMENSION = 1600
PREVIEW_MAX_DIMENSION = 960
PREVIEW_DPI = 96
PREVIEW_SCATTER_MAX_POINTS = 30000
FULL_SCATTER_MAX_POINTS = 120000
FULL_EXPORT_DPI = 180
HISTOGRAM_BINS = 192
LAB_MODEL_POINTS = 1800
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
WEB_DIR = Path(__file__).with_name("web")
MAX_CACHE_ITEMS = 24
ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
CHART_SPECS = [
    ("original", "Original"),
    ("ab", "A/B Chroma Map"),
    ("cl", "Saturation vs. Luma"),
    ("luma", "Luma Distribution + RGB Channels"),
    ("brightness_sat", "Brightness / Saturation"),
    ("sat_brightness", "Saturation / Brightness"),
]
# ─────────────────────────────────────────────────────


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_image(
    source: str | Path | io.BytesIO | io.BufferedIOBase,
    max_dimension: int = MAX_DIMENSION,
) -> np.ndarray:
    img = Image.open(source).convert("RGB")
    img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    return np.array(img) / 255.0


def sample_pixels(
    rgb_flat: np.ndarray,
    ratio: float = SAMPLE_RATIO,
    max_points: int = FULL_SCATTER_MAX_POINTS,
) -> np.ndarray:
    """随机采样像素，加快散点图渲染。"""
    n = len(rgb_flat)
    sample_size = min(n, max(5000, min(max_points, int(n * ratio))))
    idx = np.random.choice(n, size=sample_size, replace=False)
    return rgb_flat[idx]


def rgb_to_lab(rgb_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB → Lab，返回 L, a, b 通道。"""
    try:
        from skimage import color as skcolor

        lab = skcolor.rgb2lab(rgb_flat.reshape(-1, 1, 3)).reshape(-1, 3)
        return lab[:, 0], lab[:, 1], lab[:, 2]
    except ImportError:
        h, s, _ = rgb_to_hsv_components(rgb_flat)
        luma = 0.299 * rgb_flat[:, 0] + 0.587 * rgb_flat[:, 1] + 0.114 * rgb_flat[:, 2]
        a = s * np.cos(np.radians(h * 360))
        b = s * np.sin(np.radians(h * 360))
        return luma * 100, a * 100, b * 100


def rgb_to_lab_ab(rgb_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, a_vals, b_vals = rgb_to_lab(rgb_flat)
    return a_vals, b_vals


def rgb_to_hsv_components(rgb_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB flat array → H(0-1), S(0-1), V(0-1)。"""
    r, g, b = rgb_flat[:, 0], rgb_flat[:, 1], rgb_flat[:, 2]
    mx = np.max(rgb_flat, axis=1)
    mn = np.min(rgb_flat, axis=1)
    diff = mx - mn

    v = mx
    s = np.zeros_like(mx)
    np.divide(diff, mx, out=s, where=mx > 0)

    h = np.zeros(len(r))
    mask_r = (mx == r) & (diff > 0)
    mask_g = (mx == g) & (diff > 0)
    mask_b = (mx == b) & (diff > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = h / 6.0

    return h, s, v


def pixel_colors_from_rgb(rgb_flat: np.ndarray) -> np.ndarray:
    return rgb_flat.clip(0, 1)


def create_analysis_context(rgb: np.ndarray, scatter_max_points: int) -> dict[str, Any]:
    rgb_flat = rgb.reshape(-1, 3)
    h_full, s_full, v_full = rgb_to_hsv_components(rgb_flat)
    luma_full = 0.299 * rgb_flat[:, 0] + 0.587 * rgb_flat[:, 1] + 0.114 * rgb_flat[:, 2]

    rgb_sampled = sample_pixels(rgb_flat, max_points=scatter_max_points)
    h_sampled, s_sampled, v_sampled = rgb_to_hsv_components(rgb_sampled)
    l_sampled, a_sampled, b_sampled = rgb_to_lab(rgb_sampled)
    luma_sampled = 0.299 * rgb_sampled[:, 0] + 0.587 * rgb_sampled[:, 1] + 0.114 * rgb_sampled[:, 2]

    return {
        "rgb": rgb,
        "rgb_flat": rgb_flat,
        "h_full": h_full,
        "s_full": s_full,
        "v_full": v_full,
        "luma_full": luma_full,
        "rgb_sampled": rgb_sampled,
        "h_sampled": h_sampled,
        "s_sampled": s_sampled,
        "v_sampled": v_sampled,
        "l_sampled": l_sampled,
        "a_sampled": a_sampled,
        "b_sampled": b_sampled,
        "luma_sampled": luma_sampled,
    }


def compute_summary(context: dict[str, Any]) -> dict[str, Any]:
    mean_rgb = np.mean(context["rgb_flat"], axis=0)
    hue_deg = float(np.mean(context["h_full"]) * 360.0)

    return {
        "pixel_count": int(len(context["rgb_flat"])),
        "mean_brightness": round(float(np.mean(context["luma_full"])), 3),
        "mean_saturation": round(float(np.mean(context["s_full"])), 3),
        "mean_hue": round(hue_deg, 1),
        "average_color": "#{:02x}{:02x}{:02x}".format(
            *(np.clip(mean_rgb * 255, 0, 255).astype(int))
        ),
    }


def compute_lab_model(context: dict[str, Any], max_points: int = LAB_MODEL_POINTS) -> dict[str, Any]:
    rgb_flat = context["rgb_flat"]
    sample_size = min(len(rgb_flat), max_points)
    idx = np.random.choice(len(rgb_flat), size=sample_size, replace=False)
    sampled = rgb_flat[idx]
    l_vals, a_vals, b_vals = rgb_to_lab(sampled)
    colors = np.clip(sampled * 255, 0, 255).astype(int)

    points = []
    for l_val, a_val, b_val, color in zip(l_vals, a_vals, b_vals, colors):
        points.append(
            {
                "l": round(float(l_val), 2),
                "a": round(float(a_val), 2),
                "b": round(float(b_val), 2),
                "color": "#{:02x}{:02x}{:02x}".format(*color),
            }
        )

    return {
        "x_label": "B (Blue to Yellow)",
        "y_label": "A (Green to Red)",
        "z_label": "L (Dark to Light)",
        "ab_range": 110,
        "l_range": [0, 100],
        "points": points,
    }


def pca_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values ** 2
    variance_ratio = variance / variance.sum() if variance.sum() > 0 else np.array([0.0, 0.0])
    return vh, variance_ratio


def describe_color_bias(mean_a: float, mean_b: float) -> str:
    vertical = "red" if mean_a > 10 else "green" if mean_a < -10 else "neutral-a"
    horizontal = "yellow" if mean_b > 10 else "blue" if mean_b < -10 else "neutral-b"
    if vertical.startswith("neutral") and horizontal.startswith("neutral"):
        return "near-neutral"
    if vertical.startswith("neutral"):
        return horizontal
    if horizontal.startswith("neutral"):
        return vertical
    return f"{vertical}-{horizontal}"


def analyze_lab_ab_signature(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    ab_points = np.vstack([
        np.column_stack([context["a_sampled"], context["b_sampled"]])
        for context in contexts
    ])
    if len(ab_points) > 50000:
        idx = np.random.choice(len(ab_points), size=50000, replace=False)
        ab_points = ab_points[idx]
    components, variance_ratio = pca_2d(ab_points)
    vector = components[0]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    return {
        "axis_angle": round(angle, 1),
        "axis_strength": round(float(variance_ratio[0]), 3),
    }


def analyze_color_harmony(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    hue_values = np.concatenate([context["h_sampled"] * 360.0 for context in contexts])
    sat_values = np.concatenate([context["s_sampled"] for context in contexts])
    valid_hues = hue_values[sat_values > 0.16]
    if len(valid_hues) == 0:
        return {"harmony_type": "neutral", "dominant_hues": []}
    hist, edges = np.histogram(valid_hues, bins=18, range=(0, 360))
    top_bins = np.argsort(hist)[-3:][::-1]
    dominant = [round(float((edges[idx] + edges[idx + 1]) / 2), 1) for idx in top_bins if hist[idx] > 0]

    sampled = valid_hues if len(valid_hues) <= 120 else valid_hues[np.random.choice(len(valid_hues), size=120, replace=False)]
    diffs = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            diff = abs(sampled[i] - sampled[j])
            diffs.append(min(diff, 360 - diff))
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    anchors = {30: "analogous", 120: "triadic", 180: "complementary"}
    harmony_type = anchors[min(anchors, key=lambda key: abs(key - mean_diff))]
    return {
        "harmony_type": harmony_type,
        "dominant_hues": dominant,
    }


def analyze_luminance_layers(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    rgb_flat = np.vstack([context["rgb_flat"] for context in contexts])
    if len(rgb_flat) > 80000:
        idx = np.random.choice(len(rgb_flat), size=80000, replace=False)
        rgb_flat = rgb_flat[idx]
    l_vals, a_vals, b_vals = rgb_to_lab(rgb_flat)
    layers = {
        "shadow": l_vals < 33,
        "midtone": (l_vals >= 33) & (l_vals < 67),
        "highlight": l_vals >= 67,
    }
    results = {}
    for name, mask in layers.items():
        if mask.sum() == 0:
            results[name] = {"bias": "empty", "a": 0.0, "b": 0.0}
            continue
        mean_a = float(np.mean(a_vals[mask]))
        mean_b = float(np.mean(b_vals[mask]))
        results[name] = {
            "bias": describe_color_bias(mean_a, mean_b),
            "a": round(mean_a, 1),
            "b": round(mean_b, 1),
        }
    return results


def analyze_contrast(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    spatial_scores = []
    color_scores = []
    for context in contexts:
        luma_2d = 0.299 * context["rgb"][:, :, 0] + 0.587 * context["rgb"][:, :, 1] + 0.114 * context["rgb"][:, :, 2]
        dx = np.abs(np.diff(luma_2d, axis=1)).mean()
        dy = np.abs(np.diff(luma_2d, axis=0)).mean()
        spatial_scores.append(float((dx + dy) * 255))

        lab_image = np.stack([context["l_sampled"], context["a_sampled"], context["b_sampled"]], axis=1)
        if len(lab_image) > 1:
            diffs = np.sqrt(np.sum(np.diff(lab_image, axis=0) ** 2, axis=1))
            color_scores.append(float(np.mean(diffs)))

    return {
        "spatial_contrast": round(float(np.mean(spatial_scores)), 2),
        "color_contrast": round(float(np.mean(color_scores) if color_scores else 0.0), 2),
    }


def analyze_spatial_color(images_rgb: list[np.ndarray]) -> list[dict[str, Any]]:
    grid_colors = np.zeros((3, 3, 3), dtype=float)
    for rgb in images_rgb:
        height, width = rgb.shape[:2]
        for row in range(3):
            for col in range(3):
                tile = rgb[row * height // 3:(row + 1) * height // 3, col * width // 3:(col + 1) * width // 3]
                grid_colors[row, col] += tile.reshape(-1, 3).mean(axis=0)
    grid_colors /= max(1, len(images_rgb))

    cells = []
    for row in range(3):
        for col in range(3):
            color = np.clip(grid_colors[row, col] * 255, 0, 255).astype(int)
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "color": "#{:02x}{:02x}{:02x}".format(*color),
                }
            )
    return cells


def analyze_mood(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    saturation = float(np.mean([np.mean(context["s_full"]) for context in contexts]))
    brightness = float(np.mean([np.mean(context["luma_full"]) for context in contexts]))
    warmth = float(np.mean([np.mean(context["b_sampled"]) for context in contexts]))
    return {
        "warmth": round(warmth, 1),
        "saturation": round(saturation, 3),
        "brightness": round(brightness, 3),
    }


def analyze_fingerprints(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for context in contexts:
        h_hist, _ = np.histogram(context["h_full"], bins=18, range=(0, 1))
        s_hist, _ = np.histogram(context["s_full"], bins=8, range=(0, 1))
        v_hist, _ = np.histogram(context["v_full"], bins=8, range=(0, 1))
        feat = np.concatenate([h_hist, s_hist, v_hist]).astype(float)
        feat = feat / max(1.0, feat.sum())
        features.append(feat)
    features_np = np.vstack(features)
    centroid = features_np.mean(axis=0, keepdims=True)
    distances = np.sqrt(np.sum((features_np - centroid) ** 2, axis=1))
    cohesion = clamp(1.0 - float(np.mean(distances)) * 3.0, 0.0, 1.0)
    return {"cohesion": round(cohesion, 3)}


def describe_collection_style(aggregate: dict[str, Any]) -> dict[str, Any]:
    metrics = aggregate["group_metrics"]
    brightness = metrics["mean_brightness"]
    saturation = metrics["mean_saturation"]
    hue = metrics["mean_hue"]
    cohesion = aggregate["fingerprint"]["cohesion"]
    harmony = aggregate["harmony"]["harmony_type"]
    principal = aggregate["lab_signature"]["axis_strength"]

    brightness_desc = "high-key" if brightness >= 0.68 else "low-key" if brightness <= 0.38 else "balanced-light"
    saturation_desc = "muted" if saturation <= 0.28 else "vivid" if saturation >= 0.58 else "controlled-saturation"
    consistency_desc = "highly consistent" if cohesion >= 0.74 else "moderately varied" if cohesion >= 0.45 else "visibly diverse"

    if 25 <= hue <= 70:
        hue_desc = "warm amber"
    elif 70 < hue <= 160:
        hue_desc = "green-leaning"
    elif 160 < hue <= 250:
        hue_desc = "cool cyan-blue"
    elif 250 < hue <= 320:
        hue_desc = "violet-magenta"
    else:
        hue_desc = "red-orange"

    axis_desc = "with a strong directional color axis" if principal >= 0.62 else "with a broad multi-directional palette"
    summary_text = (
        f"This set reads as a {consistency_desc} visual series with {brightness_desc} tonality, "
        f"{saturation_desc} color handling, a {hue_desc} cast, and {harmony} hue relationships, {axis_desc}."
    )
    return {
        "style_tags": [brightness_desc, saturation_desc, hue_desc, harmony, consistency_desc],
        "style_summary": summary_text,
    }


def analyze_collection(images: list[dict[str, str]]) -> dict[str, Any]:
    if not images:
        raise ValueError("缺少文件夹图片数据")

    summaries = []
    previews = []
    contexts = []
    images_rgb = []
    for item in images:
        filename = item.get("filename", "upload.jpg")
        data_url = item.get("data_url")
        if not data_url:
            raise ValueError(f"{filename} 缺少图片数据")
        image_bytes = decode_data_url(data_url)
        rgb = load_image(io.BytesIO(image_bytes), max_dimension=PREVIEW_MAX_DIMENSION)
        context = create_analysis_context(rgb, scatter_max_points=PREVIEW_SCATTER_MAX_POINTS)
        contexts.append(context)
        images_rgb.append(rgb)
        summary = compute_summary(context)
        summaries.append(summary)
        previews.append(
            {
                "filename": filename,
                "preview_image": image_to_data_url(render_chart_bytes("original", context, preview=True)),
                "average_color": summary["average_color"],
                "mean_brightness": summary["mean_brightness"],
                "mean_saturation": summary["mean_saturation"],
            }
        )

    mean_brightness = float(np.mean([item["mean_brightness"] for item in summaries]))
    mean_saturation = float(np.mean([item["mean_saturation"] for item in summaries]))
    mean_hue = float(np.mean([item["mean_hue"] for item in summaries]))
    brightness_range = [float(min(item["mean_brightness"] for item in summaries)), float(max(item["mean_brightness"] for item in summaries))]
    saturation_range = [float(min(item["mean_saturation"] for item in summaries)), float(max(item["mean_saturation"] for item in summaries))]

    mean_rgb = np.mean(
        [
            [int(item["average_color"][1:3], 16), int(item["average_color"][3:5], 16), int(item["average_color"][5:7], 16)]
            for item in summaries
        ],
        axis=0,
    )

    aggregate = {
        "group_metrics": {
            "mean_brightness": round(mean_brightness, 3),
            "mean_saturation": round(mean_saturation, 3),
            "mean_hue": round(mean_hue, 1),
            "brightness_range": [round(brightness_range[0], 3), round(brightness_range[1], 3)],
            "saturation_range": [round(saturation_range[0], 3), round(saturation_range[1], 3)],
            "average_color": "#{:02x}{:02x}{:02x}".format(*np.clip(mean_rgb, 0, 255).astype(int)),
        },
        "lab_signature": analyze_lab_ab_signature(contexts),
        "harmony": analyze_color_harmony(contexts),
        "luminance_layers": analyze_luminance_layers(contexts),
        "contrast": analyze_contrast(contexts),
        "spatial_palette": analyze_spatial_color(images_rgb),
        "mood": analyze_mood(contexts),
        "fingerprint": analyze_fingerprints(contexts),
    }
    style = describe_collection_style(aggregate)
    labels = [item["filename"] for item in images]
    collection_chart_builders = [
        ("lab_ab_distribution", "Lab a*b* Distribution", lambda: build_collection_lab_ab_figure(contexts)),
        ("color_harmony", "Color Harmony", lambda: build_collection_harmony_figure(contexts)),
        ("hue_saturation_heatmap", "Hue / Saturation Heatmap", lambda: build_collection_hs_heatmap_figure(contexts)),
        ("luminance_layers", "Luminance Layers", lambda: build_collection_luminance_layers_figure(contexts)),
        ("contrast_profile", "Contrast Profile", lambda: build_collection_contrast_figure(contexts, labels)),
        ("spatial_palette", "Spatial Palette Grid", lambda: build_collection_spatial_palette_figure(aggregate["spatial_palette"])),
        ("fingerprint_map", "Image Fingerprint Map", lambda: build_collection_fingerprint_figure(contexts, labels)),
        ("mood_coordinates", "Mood Coordinates", lambda: build_collection_mood_figure(contexts, labels)),
    ]
    collection_charts = []
    for chart_id, title, builder in collection_chart_builders:
        collection_charts.append(
            {
                "id": chart_id,
                "title": title,
                "image": image_to_data_url(figure_to_png_bytes(builder(), dpi=PREVIEW_DPI)),
                "full_image": image_to_data_url(figure_to_png_bytes(builder(), dpi=FULL_EXPORT_DPI)),
            }
        )
    return {
        "image_count": len(images),
        "collection_summary": style["style_summary"],
        "style_tags": style["style_tags"],
        **aggregate,
        "charts": collection_charts,
        "images": previews,
    }


def plot_ab_chart(ax: plt.Axes, context: dict[str, Any]) -> None:
    ax.scatter(
        context["b_sampled"],
        context["a_sampled"],
        c=pixel_colors_from_rgb(context["rgb_sampled"]),
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        rasterized=True,
    )
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_facecolor("#1a1a1a")
    ax.set_xlabel("B (Blue to Yellow)", color="white", fontsize=8)
    ax.set_ylabel("A (Green to Red)", color="white", fontsize=8)
    ax.set_title("A/B Chroma Map", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.text(88, 5, "Yellow", color="yellow", fontsize=7, alpha=0.7)
    ax.text(-105, 5, "Blue", color="#88aaff", fontsize=7, alpha=0.7)
    ax.text(5, 95, "Red", color="#ff8888", fontsize=7, alpha=0.7)
    ax.text(5, -105, "Green", color="#88ff88", fontsize=7, alpha=0.7)


def plot_cl_chart(ax: plt.Axes, context: dict[str, Any]) -> None:
    ax.scatter(
        context["luma_sampled"],
        context["s_sampled"],
        c=pixel_colors_from_rgb(context["rgb_sampled"]),
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        rasterized=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a1a")
    ax.set_xlabel("Luma (Dark to Bright)", color="white", fontsize=8)
    ax.set_ylabel("Saturation", color="white", fontsize=8)
    ax.set_title("Saturation vs. Luma", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def plot_luma_sat_chart(ax: plt.Axes, context: dict[str, Any]) -> None:
    bins = np.linspace(0, 1, 33)
    centers = (bins[:-1] + bins[1:]) / 2
    mean_sat = []
    for i in range(len(bins) - 1):
        mask = (context["luma_sampled"] >= bins[i]) & (context["luma_sampled"] < bins[i + 1])
        mean_sat.append(context["s_sampled"][mask].mean() if mask.sum() > 0 else 0)

    colors = plt.cm.plasma(centers)
    ax.bar(centers, mean_sat, width=(bins[1] - bins[0]) * 0.9, color=colors, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a1a")
    ax.set_xlabel("Brightness Range", color="white", fontsize=8)
    ax.set_ylabel("Average Saturation", color="white", fontsize=8)
    ax.set_title("Brightness / Saturation", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def plot_sat_luma_chart(ax: plt.Axes, context: dict[str, Any]) -> None:
    bins = np.linspace(0, 1, 33)
    centers = (bins[:-1] + bins[1:]) / 2
    mean_luma = []
    for i in range(len(bins) - 1):
        mask = (context["s_sampled"] >= bins[i]) & (context["s_sampled"] < bins[i + 1])
        mean_luma.append(context["luma_sampled"][mask].mean() if mask.sum() > 0 else 0)

    colors = plt.cm.viridis(centers)
    ax.bar(centers, mean_luma, width=(bins[1] - bins[0]) * 0.9, color=colors, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a1a")
    ax.set_xlabel("Saturation Range", color="white", fontsize=8)
    ax.set_ylabel("Average Brightness", color="white", fontsize=8)
    ax.set_title("Saturation / Brightness", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def plot_luminance_chart(ax: plt.Axes, context: dict[str, Any]) -> None:
    ax.hist(context["luma_full"], bins=HISTOGRAM_BINS, color="#aaaaaa", alpha=0.7, density=True)
    for ch, col in zip([0, 1, 2], ["#ff4444", "#44ff44", "#4488ff"]):
        ax.hist(context["rgb_flat"][:, ch], bins=HISTOGRAM_BINS, color=col, alpha=0.35, density=True)

    ax.set_xlim(0, 1)
    ax.set_facecolor("#1a1a1a")
    ax.set_xlabel("Brightness", color="white", fontsize=8)
    ax.set_ylabel("Density", color="white", fontsize=8)
    ax.set_title("Luma Distribution + RGB Channels", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def style_chart_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.set_facecolor("#1a1a1a")


def build_original_figure(context: dict[str, Any], title: str, size: tuple[float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, facecolor="#111111")
    ax.imshow(context["rgb"])
    ax.axis("off")
    style_chart_axis(ax, title)
    fig.tight_layout(pad=0.6)
    return fig


def build_ab_figure(context: dict[str, Any], size: tuple[float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, facecolor="#111111")
    plot_ab_chart(ax, context)
    fig.tight_layout(pad=1.2)
    return fig


def build_cl_figure(context: dict[str, Any], size: tuple[float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, facecolor="#111111")
    plot_cl_chart(ax, context)
    fig.tight_layout(pad=1.2)
    return fig


def build_luma_sat_figure(context: dict[str, Any], size: tuple[float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, facecolor="#111111")
    plot_luma_sat_chart(ax, context)
    fig.tight_layout(pad=1.2)
    return fig


def build_sat_luma_figure(context: dict[str, Any], size: tuple[float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, facecolor="#111111")
    plot_sat_luma_chart(ax, context)
    fig.tight_layout(pad=1.2)
    return fig


def build_luminance_figure(context: dict[str, Any], size: tuple[float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, facecolor="#111111")
    plot_luminance_chart(ax, context)
    fig.tight_layout(pad=1.2)
    return fig


def figure_to_png_bytes(fig: plt.Figure, dpi: int) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=dpi, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return buffer.getvalue()


def build_chart_figure(chart_id: str, context: dict[str, Any], preview: bool) -> plt.Figure:
    large_square = (8, 8) if not preview else (4.2, 4.2)
    wide = (8, 6) if not preview else (4.6, 3.4)
    if chart_id == "original":
        return build_original_figure(context, "Original", large_square)
    if chart_id == "ab":
        return build_ab_figure(context, large_square)
    if chart_id == "cl":
        return build_cl_figure(context, wide)
    if chart_id == "luma":
        return build_luminance_figure(context, wide)
    if chart_id == "brightness_sat":
        return build_luma_sat_figure(context, wide)
    if chart_id == "sat_brightness":
        return build_sat_luma_figure(context, wide)
    raise ValueError(f"未知图表: {chart_id}")


def render_chart_bytes(chart_id: str, context: dict[str, Any], preview: bool) -> bytes:
    fig = build_chart_figure(chart_id, context, preview=preview)
    dpi = PREVIEW_DPI if preview else FULL_EXPORT_DPI
    return figure_to_png_bytes(fig, dpi=dpi)


def build_collection_lab_ab_figure(contexts: list[dict[str, Any]]) -> plt.Figure:
    points = np.vstack([
        np.column_stack([context["a_sampled"], context["b_sampled"]])
        for context in contexts
    ])
    colors = np.vstack([context["rgb_sampled"] for context in contexts])
    if len(points) > 50000:
        idx = np.random.choice(len(points), size=50000, replace=False)
        points = points[idx]
        colors = colors[idx]

    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="#111111")
    ax.scatter(points[:, 1], points[:, 0], c=colors, s=1.6, alpha=0.28, rasterized=True)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.24)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.24)
    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_xlabel("b* (Blue to Yellow)", color="white", fontsize=8)
    ax.set_ylabel("a* (Green to Red)", color="white", fontsize=8)
    ax.set_title("Collection Lab a*b* Distribution", color="white", fontsize=10)
    ax.set_facecolor("#151515")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout(pad=1)
    return fig


def build_collection_hs_heatmap_figure(contexts: list[dict[str, Any]]) -> plt.Figure:
    hues = np.concatenate([context["h_full"] * 360.0 for context in contexts])
    sats = np.concatenate([context["s_full"] for context in contexts])
    heatmap, xedges, yedges = np.histogram2d(hues, sats, bins=[72, 36], range=[[0, 360], [0, 1]])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="#111111")
    ax.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        extent=[0, 360, 0, 1],
        cmap="magma",
    )
    ax.set_xlabel("Hue (degrees)", color="white", fontsize=8)
    ax.set_ylabel("Saturation", color="white", fontsize=8)
    ax.set_title("Hue / Saturation Heatmap", color="white", fontsize=10)
    ax.set_facecolor("#151515")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout(pad=1)
    return fig


def build_collection_harmony_figure(contexts: list[dict[str, Any]]) -> plt.Figure:
    hue_values = np.concatenate([context["h_sampled"] * 360.0 for context in contexts])
    sat_values = np.concatenate([context["s_sampled"] for context in contexts])
    valid_hues = hue_values[sat_values > 0.16]
    if len(valid_hues) == 0:
        valid_hues = hue_values

    sampled = valid_hues
    if len(sampled) > 140:
        idx = np.random.choice(len(sampled), size=140, replace=False)
        sampled = sampled[idx]

    diffs = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            diff = abs(sampled[i] - sampled[j])
            diffs.append(min(diff, 360 - diff))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), facecolor="#111111")
    axes[0].hist(valid_hues, bins=36, color="#ad6a44", alpha=0.92)
    axes[0].set_title("Dominant Hue Distribution", color="white", fontsize=10)
    axes[0].set_xlabel("Hue (degrees)", color="white", fontsize=8)
    axes[0].set_ylabel("Count", color="white", fontsize=8)

    axes[1].hist(diffs, bins=36, color="#4f8c8d", alpha=0.92)
    for x_val, label in [(30, "Analogous"), (120, "Triadic"), (180, "Complementary")]:
        axes[1].axvline(x_val, color="#ffb36d", linestyle="--", linewidth=0.9, alpha=0.8)
        axes[1].text(x_val, max(axes[1].get_ylim()[1] * 0.9, 1), label, color="#ffd9b0", fontsize=7, ha="center")
    axes[1].set_title("Hue Angle Differences", color="white", fontsize=10)
    axes[1].set_xlabel("Angle Difference", color="white", fontsize=8)
    axes[1].set_ylabel("Count", color="white", fontsize=8)

    for ax in axes:
        ax.set_facecolor("#151515")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.tight_layout(pad=1)
    return fig


def build_collection_luminance_layers_figure(contexts: list[dict[str, Any]]) -> plt.Figure:
    rgb_flat = np.vstack([context["rgb_flat"] for context in contexts])
    if len(rgb_flat) > 90000:
        idx = np.random.choice(len(rgb_flat), size=90000, replace=False)
        rgb_flat = rgb_flat[idx]

    l_vals, a_vals, b_vals = rgb_to_lab(rgb_flat)
    colors = pixel_colors_from_rgb(rgb_flat)
    layers = [
        ("Shadow", l_vals < 33),
        ("Midtone", (l_vals >= 33) & (l_vals < 67)),
        ("Highlight", l_vals >= 67),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), facecolor="#111111")
    for ax, (title, mask) in zip(axes, layers):
        layer_count = int(mask.sum())
        if layer_count:
            layer_idx = np.where(mask)[0]
            if layer_count > 10000:
                layer_idx = np.random.choice(layer_idx, size=10000, replace=False)
            ax.scatter(
                b_vals[layer_idx],
                a_vals[layer_idx],
                c=colors[layer_idx],
                s=2.4,
                alpha=0.3,
                rasterized=True,
            )
            mean_a = float(np.mean(a_vals[layer_idx]))
            mean_b = float(np.mean(b_vals[layer_idx]))
            ax.plot(mean_b, mean_a, marker="+", markersize=18, markeredgewidth=2.8, color="white")
            subtitle = f"a={mean_a:.1f}, b={mean_b:.1f}"
        else:
            subtitle = "No sampled pixels"

        ax.axhline(0, color="white", linewidth=0.5, alpha=0.24)
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.24)
        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)
        ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=9)
        ax.set_xlabel("B (Blue to Yellow)", color="white", fontsize=8)
        ax.set_ylabel("A (Green to Red)", color="white", fontsize=8)
        ax.set_facecolor("#151515")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.tight_layout(pad=1)
    return fig


def build_collection_contrast_figure(contexts: list[dict[str, Any]], labels: list[str]) -> plt.Figure:
    spatial_scores = []
    color_scores = []
    short_labels = []
    for context, label in zip(contexts, labels):
        luma_2d = 0.299 * context["rgb"][:, :, 0] + 0.587 * context["rgb"][:, :, 1] + 0.114 * context["rgb"][:, :, 2]
        dx = np.abs(np.diff(luma_2d, axis=1)).mean()
        dy = np.abs(np.diff(luma_2d, axis=0)).mean()
        spatial_scores.append(float((dx + dy) * 255))

        lab_points = np.stack([context["l_sampled"], context["a_sampled"], context["b_sampled"]], axis=1)
        if len(lab_points) > 1:
            diffs = np.sqrt(np.sum(np.diff(lab_points, axis=0) ** 2, axis=1))
            color_scores.append(float(np.mean(diffs)))
        else:
            color_scores.append(0.0)
        short_labels.append(Path(label).name[:12])

    positions = np.arange(len(short_labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.8, len(short_labels) * 0.7), 4.2), facecolor="#111111")
    ax.bar(positions - width / 2, spatial_scores, width=width, color="#c68159", alpha=0.9, label="Spatial")
    ax.bar(positions + width / 2, color_scores, width=width, color="#5f8d8f", alpha=0.9, label="Color")
    ax.set_title("Per-image Contrast Profile", color="white", fontsize=10)
    ax.set_xlabel("Image", color="white", fontsize=8)
    ax.set_ylabel("Score", color="white", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, rotation=35, ha="right")
    ax.legend(facecolor="#151515", edgecolor="#444", labelcolor="white")
    ax.set_facecolor("#151515")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout(pad=1)
    return fig


def build_collection_spatial_palette_figure(cells: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="#111111")
    for cell in cells:
        row, col = cell["row"], cell["col"]
        ax.add_patch(plt.Rectangle((col, 2 - row), 1, 1, color=cell["color"]))
        ax.text(
            col + 0.5,
            2 - row + 0.5,
            cell["color"],
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.28, "edgecolor": "none"},
        )
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Spatial Palette Grid", color="white", fontsize=10, pad=10)
    fig.tight_layout(pad=1)
    return fig


def build_collection_fingerprint_figure(contexts: list[dict[str, Any]], labels: list[str]) -> plt.Figure:
    features = []
    for context in contexts:
        h_hist, _ = np.histogram(context["h_full"], bins=18, range=(0, 1))
        s_hist, _ = np.histogram(context["s_full"], bins=8, range=(0, 1))
        v_hist, _ = np.histogram(context["v_full"], bins=8, range=(0, 1))
        feat = np.concatenate([h_hist, s_hist, v_hist]).astype(float)
        feat = feat / max(1.0, feat.sum())
        features.append(feat)
    matrix = np.vstack(features)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T

    fig, ax = plt.subplots(figsize=(6.6, 5.2), facecolor="#111111")
    ax.scatter(coords[:, 0], coords[:, 1], s=52, color="#d7a17c", alpha=0.82)
    for idx, label in enumerate(labels):
        ax.text(coords[idx, 0], coords[idx, 1], Path(label).name[:14], color="white", fontsize=7, alpha=0.82)
    ax.set_title("Image Fingerprint Map", color="white", fontsize=10)
    ax.set_xlabel("PC1", color="white", fontsize=8)
    ax.set_ylabel("PC2", color="white", fontsize=8)
    ax.set_facecolor("#151515")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout(pad=1)
    return fig


def build_collection_mood_figure(contexts: list[dict[str, Any]], labels: list[str]) -> plt.Figure:
    coords = []
    trimmed_labels = []
    for context, label in zip(contexts, labels):
        coords.append(
            [
                float(np.mean(context["s_full"])),
                float(np.mean(context["luma_full"])),
                float(np.mean(context["b_sampled"])),
            ]
        )
        trimmed_labels.append(Path(label).name[:12])

    matrix = np.array(coords)
    fig, ax = plt.subplots(figsize=(6.6, 5.2), facecolor="#111111")
    scatter = ax.scatter(
        matrix[:, 2],
        matrix[:, 0],
        c=matrix[:, 1],
        cmap="viridis",
        s=120,
        alpha=0.86,
        edgecolors="#f6f2ea",
        linewidths=0.5,
    )
    for idx, label in enumerate(trimmed_labels):
        ax.text(matrix[idx, 2], matrix[idx, 0], label, color="white", fontsize=7, alpha=0.84)
    ax.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.set_title("Mood Coordinates", color="white", fontsize=10)
    ax.set_xlabel("Warmth / Coolness", color="white", fontsize=8)
    ax.set_ylabel("Average Saturation", color="white", fontsize=8)
    ax.set_facecolor("#151515")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="gray")
    plt.setp(cbar.ax.get_yticklabels(), color="gray", fontsize=7)
    cbar.set_label("Average Brightness", color="white", fontsize=8)
    fig.tight_layout(pad=1)
    return fig


def analyze_rgb_image(rgb: np.ndarray, title: str, job_id: str | None = None) -> dict[str, Any]:
    preview_context = create_analysis_context(rgb, scatter_max_points=PREVIEW_SCATTER_MAX_POINTS)

    charts = []
    for chart_id, chart_title in CHART_SPECS:
        preview_png = render_chart_bytes(chart_id, preview_context, preview=True)
        chart_payload = {
            "id": chart_id,
            "title": chart_title,
            "image": image_to_data_url(preview_png),
        }
        if job_id is not None:
            chart_payload["full_url"] = f"/chart?job={job_id}&chart={chart_id}"
        charts.append(chart_payload)

    return {
        "summary": compute_summary(preview_context),
        "lab_model": compute_lab_model(preview_context),
        "charts": charts,
    }


def analyze_image_file(img_path: str | Path, output_dir: str | Path) -> str:
    img_path = Path(img_path)
    output_dir = Path(output_dir)
    print(f"  处理: {img_path.name}")

    rgb = load_image(img_path, max_dimension=MAX_DIMENSION)
    full_context = create_analysis_context(rgb, scatter_max_points=FULL_SCATTER_MAX_POINTS)
    image_dir = output_dir / img_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    for chart_id, _ in CHART_SPECS:
        out_path = image_dir / f"{chart_id}.png"
        out_path.write_bytes(render_chart_bytes(chart_id, full_context, preview=False))
    print(f"  → 保存: {image_dir}")
    return str(image_dir)


def image_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("无效的图片数据")
    return base64.b64decode(data_url.split(",", 1)[1])


def read_text_file(path: Path) -> bytes:
    return path.read_bytes()


def store_job(image_bytes: bytes, filename: str) -> str:
    while len(ANALYSIS_CACHE) >= MAX_CACHE_ITEMS:
        ANALYSIS_CACHE.pop(next(iter(ANALYSIS_CACHE)))
    job_id = uuid4().hex
    ANALYSIS_CACHE[job_id] = {"image_bytes": image_bytes, "filename": filename}
    return job_id


def get_job(job_id: str) -> dict[str, Any]:
    job = ANALYSIS_CACHE.get(job_id)
    if not job:
        raise KeyError("分析结果已过期，请重新上传图片")
    return job


class ColorAnalyzerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_bytes(HTTPStatus.OK, read_text_file(WEB_DIR / "index.html"), "text/html; charset=utf-8")
            return
        if parsed.path == "/collection.html":
            self._send_bytes(HTTPStatus.OK, read_text_file(WEB_DIR / "collection.html"), "text/html; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._send_bytes(HTTPStatus.OK, read_text_file(WEB_DIR / "styles.css"), "text/css; charset=utf-8")
            return
        if parsed.path == "/chart":
            try:
                params = parse_qs(parsed.query)
                job_id = params.get("job", [None])[0]
                chart_id = params.get("chart", [None])[0]
                if not job_id or not chart_id:
                    raise ValueError("缺少图表参数")
                job = get_job(job_id)
                rgb = load_image(io.BytesIO(job["image_bytes"]), max_dimension=MAX_DIMENSION)
                context = create_analysis_context(rgb, scatter_max_points=FULL_SCATTER_MAX_POINTS)
                chart_png = render_chart_bytes(chart_id, context, preview=False)
                self._send_bytes(HTTPStatus.OK, chart_png, "image/png")
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in {"/analyze", "/analyze-collection"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
            if self.path == "/analyze-collection":
                images = payload.get("images")
                if not images:
                    raise ValueError("缺少文件夹图片数据")
                self._send_json(HTTPStatus.OK, analyze_collection(images))
                return

            images = payload.get("images")
            if not images:
                filename = payload.get("filename", "upload.jpg")
                data_url = payload.get("data_url")
                if not data_url:
                    raise ValueError("缺少图片数据")
                images = [{"filename": filename, "data_url": data_url}]

            results = []
            for item in images:
                filename = item.get("filename", "upload.jpg")
                data_url = item.get("data_url")
                if not data_url:
                    raise ValueError(f"{filename} 缺少图片数据")

                image_bytes = decode_data_url(data_url)
                job_id = store_job(image_bytes, filename)
                rgb = load_image(io.BytesIO(image_bytes), max_dimension=PREVIEW_MAX_DIMENSION)
                analysis = analyze_rgb_image(rgb, filename, job_id=job_id)
                results.append(
                    {
                        "job_id": job_id,
                        "filename": filename,
                        "summary": analysis["summary"],
                        "lab_model": analysis["lab_model"],
                        "charts": analysis["charts"],
                    }
                )

            self._send_json(HTTPStatus.OK, {"results": results})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")


def run_batch(image_dir: str | Path) -> None:
    image_dir = Path(image_dir)
    output_dir = image_dir / "analysis_output"
    output_dir.mkdir(exist_ok=True)

    jpgs = [f for f in sorted(image_dir.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg"}][:MAX_IMAGES]
    if not jpgs:
        print(f"在 {image_dir} 中没有找到 JPG 图片")
        return

    print(f"找到 {len(jpgs)} 张图片，开始分析...\n")
    for img_path in jpgs:
        analyze_image_file(img_path, output_dir)
    print(f"\n完成！结果保存在: {output_dir}")


def run_web_app(host: str = WEB_HOST, port: int = WEB_PORT) -> None:
    if not WEB_DIR.exists():
        raise FileNotFoundError(f"缺少网页资源目录: {WEB_DIR}")

    server = ThreadingHTTPServer((host, port), ColorAnalyzerHandler)
    print(f"网页应用已启动: http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图片色彩分析工具")
    parser.add_argument("image_dir", nargs="?", default=IMAGE_DIR, help="命令行批量分析时的图片目录")
    parser.add_argument("--web", action="store_true", help="启动本地网页应用")
    parser.add_argument("--host", default=WEB_HOST, help="网页服务监听地址")
    parser.add_argument("--port", type=int, default=WEB_PORT, help="网页服务端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.web:
        run_web_app(host=args.host, port=args.port)
        return
    run_batch(args.image_dir)


if __name__ == "__main__":
    main()
