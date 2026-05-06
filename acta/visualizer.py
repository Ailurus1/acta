from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("_") or "layer"


def _pretty_layer_name(name: str) -> str:
    parts = name.split(".")
    for i in range(len(parts) - 2):
        if parts[i] == "transformer" and parts[i + 1] == "h":
            try:
                n = int(parts[i + 2])
                return f"layer_{n}"
            except ValueError:
                pass
            break
    return ".".join(parts)


def _upsample_grid(z: np.ndarray, up_x: int = 6, up_y: int = 4) -> np.ndarray:
    if up_x <= 1 and up_y <= 1:
        return z
    y, x = z.shape
    x_old = np.arange(x)
    x_new = np.linspace(0, x - 1, x * up_x)
    z_x = np.vstack([np.interp(x_new, x_old, z_row) for z_row in z])
    y_old = np.arange(y)
    y_new = np.linspace(0, y - 1, y * up_y)
    z_xy = np.vstack([np.interp(y_new, y_old, z_x[:, j]) for j in range(z_x.shape[1])]).T
    return z_xy

def _plot_token_trends(outliers: dict[str, Any], output_dir: Path) -> None:
    token_trends = outliers.get("token_trends")
    if not token_trends:
        return

    layer_names = token_trends.get("layer_names", [])
    means = token_trends.get("mean", [])
    variances = token_trends.get("variance", [])
    token_count = int(token_trends.get("token_count", 0))
    tokens = token_trends.get("tokens", None)
    if not layer_names or not means or not variances or token_count <= 0:
        return

    # means/variances are [num_layers][token_count]
    num_layers = len(layer_names)
    if len(means) != num_layers or len(variances) != num_layers:
        return

    trend_dir = output_dir / "token_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(num_layers)
    for token_idx in range(token_count):
        y = np.array([float(means[i][token_idx]) for i in range(num_layers)], dtype=np.float32)
        v = np.array([float(variances[i][token_idx]) for i in range(num_layers)], dtype=np.float32)
        std = np.sqrt(np.maximum(v, 0.0))

        tok_label = None
        if isinstance(tokens, list) and token_idx < len(tokens):
            tok_label = tokens[token_idx]

        plt.figure(figsize=(max(12, num_layers * 0.4), 5))
        sns.lineplot(x=x, y=y, marker="o", linewidth=2.0, color="#3366cc")
        plt.fill_between(x, y - std, y + std, alpha=0.2, color="#6699ff", label="mean +/- std")
        plt.xticks(x, [_pretty_layer_name(n) for n in layer_names], rotation=45, ha="right")
        plt.xlabel("Layer name")
        plt.ylabel("Token mean activation (over hidden dims)")
        title = f"Token idx {token_idx}: mean activation across layers"
        if tok_label is not None:
            title = f"Token '{tok_label}': mean activation across layers"
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(trend_dir / f"token_{token_idx}.png", dpi=150)
        plt.close()


def _plot_outlier_token_feature_3d(outliers: dict[str, Any], output_dir: Path) -> None:
    tokens = outliers.get("prompt_tokens", [])
    feat_dims = outliers.get("outlier_feature_dims", [])
    magnitudes = outliers.get("token_feature_magnitude", None)  # [T][F]
    if not tokens or not feat_dims or magnitudes is None:
        return

    t = len(tokens)
    f = len(feat_dims)
    if t == 0 or f == 0:
        return
    if len(magnitudes) != t or any(len(row) != f for row in magnitudes):
        return

    mags = np.array(magnitudes, dtype=np.float32)  # [T, F]

    max_per_feat = mags.max(axis=0)  # [F]
    topk = min(3, f)
    top_feat_idx = np.argsort(-max_per_feat)[:topk]
    top_feat_idx = np.sort(top_feat_idx)  # stable left-to-right

    feat_dims = [feat_dims[int(i)] for i in top_feat_idx]
    mags = mags[:, top_feat_idx]  # [T, topk]
    f = len(feat_dims)

    mags_up = _upsample_grid(mags.T)  # [F_up, T_up]
    f_up, t_up = mags_up.shape

    fig = plt.figure(figsize=(max(12, t * 0.6), 7))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.arange(t_up), np.arange(f_up))
    ax.plot_wireframe(X, Y, mags_up, rstride=1, cstride=1, color="royalblue", linewidth=1.5)

    ax.set_xticks(np.linspace(0, t_up - 1, t))
    ax.set_xticklabels(tokens, rotation=50, ha="right")
    ax.set_yticks(np.linspace(0, f_up - 1, f))
    ax.set_yticklabels([str(d) for d in feat_dims])
    ax.set_xlabel("Token", labelpad=25)
    ax.set_ylabel("Feature dim")
    ax.set_zlabel("Magnitude (abs max over layers)")
    ax.tick_params(axis="x", which="major", pad=-4)
    ax.tick_params(axis="y", which="major", pad=-5)
    ax.tick_params(axis="z", which="major", pad=-1)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")
    plt.title("Outlier feature magnitudes by token")
    plt.tight_layout()
    plt.savefig(output_dir / "outlier_token_feature_3d.png", dpi=200)
    plt.close()


def _plot_outlier_token_feature_3d_per_layer(outliers: dict[str, Any], output_dir: Path) -> None:
    tokens = outliers.get("prompt_tokens", [])
    feat_dims = outliers.get("outlier_feature_dims", [])
    mags_by_layer = outliers.get("token_feature_magnitude_by_layer", None)  # layer -> [T][F]
    if not tokens or not feat_dims or not isinstance(mags_by_layer, dict) or not mags_by_layer:
        return

    t = len(tokens)
    f = len(feat_dims)
    if t == 0 or f == 0:
        return

    per_layer_dir = output_dir / "outlier_token_feature_3d_per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, magnitudes in mags_by_layer.items():
        if len(magnitudes) != t or any(len(row) != f for row in magnitudes):
            continue

        mags_tf = np.array(magnitudes, dtype=np.float32)  # [T, F]
        max_per_feat = mags_tf.max(axis=0)
        topk = min(3, f)
        top_feat_idx = np.argsort(-max_per_feat)[:topk]
        top_feat_idx = np.sort(top_feat_idx)
        feat_dims_top = [feat_dims[int(i)] for i in top_feat_idx]
        mags_tf = mags_tf[:, top_feat_idx]  # [T, topk]

        mags_up = _upsample_grid(mags_tf.T)  # [F_up, T_up]
        f_up, t_up = mags_up.shape

        pretty_layer = _pretty_layer_name(layer_name)
        fname_base = _sanitize_filename(pretty_layer)

        fig = plt.figure(figsize=(max(12, t * 0.6), 7))
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(t_up), np.arange(f_up))
        ax.plot_wireframe(X, Y, mags_up, rstride=1, cstride=1, color="royalblue", linewidth=1.5)
        ax.set_xlabel("Token", labelpad=25)
        ax.set_ylabel("Feature dim")
        ax.set_zlabel("Magnitude (abs)")
        ax.set_xticks(np.linspace(0, t_up - 1, t))
        ax.set_xticklabels(tokens, rotation=50, ha="right")
        ax.set_yticks(np.linspace(0, f_up - 1, len(feat_dims_top)))
        ax.set_yticklabels([str(d) for d in feat_dims_top])
        ax.tick_params(axis="x", which="major", pad=-4)
        ax.tick_params(axis="y", which="major", pad=-5)
        ax.tick_params(axis="z", which="major", pad=-1)
        plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")
        plt.title(f"{pretty_layer}: outlier feature magnitudes")
        plt.tight_layout()
        plt.savefig(per_layer_dir / f"{fname_base}.png", dpi=200)
        plt.close()


def _plot_token_layer_3d(outliers: dict[str, Any], output_dir: Path) -> None:
    trends = outliers.get("token_layer_trends", None)
    if not isinstance(trends, dict):
        return

    tokens = trends.get("tokens", [])
    layer_names = [_pretty_layer_name(n) for n in trends.get("layer_names", [])]
    max_abs = trends.get("max_abs", None)  # [L][T]
    if not tokens or not layer_names or max_abs is None:
        return

    t = len(tokens)
    l = len(layer_names)
    if t == 0 or l == 0:
        return
    if len(max_abs) != l or any(len(row) != t for row in max_abs):
        return

    mags = np.array(max_abs, dtype=np.float32)  # [L, T]

    thr = float(outliers.get("threshold", 6.0))
    keep = np.where((mags >= thr).any(axis=1))[0]
    if keep.size == 0:
        keep = np.arange(l)
    mags = mags[keep, :]
    layer_names = [layer_names[int(i)] for i in keep]
    l = len(layer_names)

    mags_up = _upsample_grid(mags, up_x=6, up_y=2)  # [L_up, T_up]
    l_up, t_up = mags_up.shape
    fig = plt.figure(figsize=(max(12, t * 0.6), max(7, l * 0.35)))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.arange(t_up), np.arange(l_up))
    ax.plot_wireframe(X, Y, mags_up, rstride=1, cstride=1, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Token", labelpad=25)
    ax.set_ylabel("Layer")
    ax.set_zlabel("Max |activation| over hidden dims")
    ax.set_xticks(np.linspace(0, t_up - 1, t))
    ax.set_xticklabels(tokens, rotation=50, ha="right")
    ax.set_yticks(np.linspace(0, l_up - 1, l))
    ax.set_yticklabels(layer_names)
    ax.tick_params(axis="x", which="major", pad=-4)
    ax.tick_params(axis="y", which="major", pad=-5)
    ax.tick_params(axis="z", which="major", pad=-1)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")
    plt.title("Per-token max activation magnitude across layers")
    plt.tight_layout()
    plt.savefig(output_dir / "token_layer_maxabs_3d.png", dpi=200)
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
    outliers = stats.get("outliers", {})

    sns.set_theme(style="whitegrid")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(outliers, dict):
        _plot_token_trends(outliers, out)
        _plot_outlier_token_feature_3d(outliers, out)
        _plot_outlier_token_feature_3d_per_layer(outliers, out)
        _plot_token_layer_3d(outliers, out)
    _plot_layer_channel_hist(layers, out)
