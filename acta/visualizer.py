from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("_") or "layer"


def _plot_channel_trends(layers: dict[str, Any], output_dir: Path) -> None:
    layer_names = list(layers.keys())
    if not layer_names:
        return

    max_channels = max((len(layers[name]["mean"]) for name in layer_names), default=0)
    if max_channels == 0:
        return

    trend_dir = output_dir / "channel_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(layer_names))
    for channel_idx in range(max_channels):
        means: list[float] = []
        stds: list[float] = []
        shown_layer_names: list[str] = []
        shown_x: list[int] = []

        for i, layer_name in enumerate(layer_names):
            layer_mean = layers[layer_name]["mean"]
            layer_var = layers[layer_name]["variance"]
            if channel_idx >= len(layer_mean):
                continue
            means.append(float(layer_mean[channel_idx]))
            stds.append(float(np.sqrt(max(layer_var[channel_idx], 0.0))))
            shown_layer_names.append(layer_name)
            shown_x.append(int(x[i]))

        if not means:
            continue

        plt.figure(figsize=(max(12, len(shown_layer_names) * 0.8), 5))
        sns.lineplot(x=shown_x, y=means, marker="o", linewidth=2.0, color="#3366cc")
        y_low = np.array(means) - np.array(stds)
        y_high = np.array(means) + np.array(stds)
        plt.fill_between(shown_x, y_low, y_high, alpha=0.2, color="#6699ff", label="mean +/- std")
        plt.xticks(shown_x, shown_layer_names, rotation=45, ha="right")
        plt.xlabel("Layer name")
        plt.ylabel("Mean activation")
        plt.title(f"Channel {channel_idx}: mean activation across layers")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(trend_dir / f"channel_{channel_idx}.png", dpi=150)
        plt.close()


def _plot_layer_channel_hist(layers: dict[str, Any], output_dir: Path) -> None:
    hist_dir = output_dir / "layer_channel_hist"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, layer_stats in layers.items():
        means = layer_stats.get("mean", [])
        if not means:
            continue

        channel_idx = np.arange(len(means))
        plt.figure(figsize=(max(10, len(means) * 0.2), 5))
        sns.barplot(x=channel_idx, y=np.array(means, dtype=np.float32), color="#44aa99")
        plt.xlabel("Channel index in layer")
        plt.ylabel("Mean activation")
        plt.title(f"{layer_name}: channel mean activations")
        plt.tight_layout()
        plt.savefig(hist_dir / f"{_sanitize_filename(layer_name)}.png", dpi=150)
        plt.close()


def draw_activation_charts(stats: dict[str, Any], output_dir: str) -> None:
    layers = stats.get("layers", {})
    if not layers:
        return

    sns.set_theme(style="whitegrid")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_channel_trends(layers, out)
    _plot_layer_channel_hist(layers, out)
