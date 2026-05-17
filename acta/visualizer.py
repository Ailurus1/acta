from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_OPERATOR_TOP_KEYS: tuple[str, ...] = (
    "top_1",
    "top_2",
    "top_3",
    "top_10",
    "top_p01",
    "top_p10",
    "top_p50",
    "top_p90",
    "top_p99",
)
_OPERATOR_TOP_LABELS: dict[str, str] = {
    "top_1": "Top-1",
    "top_2": "Top-2",
    "top_3": "Top-3",
    "top_10": "Top-10",
    "top_p01": "Top-1%",
    "top_p10": "Top-10%",
    "top_p50": "Top-50%",
    "top_p90": "Top-90%",
    "top_p99": "Top-99%",
}


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


def _upsample_grid_numpy(z: np.ndarray, up_x: int, up_y: int) -> np.ndarray:
    if up_x <= 1 and up_y <= 1:
        return z
    y_sz, x_sz = z.shape
    xp = np.arange(x_sz, dtype=np.float64)
    xi = np.linspace(0.0, float(x_sz - 1), x_sz * up_x)
    z_x = np.empty((y_sz, xi.shape[0]), dtype=z.dtype)
    for i in range(y_sz):
        z_x[i] = np.interp(xi, xp, z[i].astype(np.float64, copy=False))

    yp = np.arange(y_sz, dtype=np.float64)
    yi = np.linspace(0.0, float(y_sz - 1), y_sz * up_y)
    z_xy = np.empty((yi.shape[0], z_x.shape[1]), dtype=z.dtype)
    for j in range(z_x.shape[1]):
        z_xy[:, j] = np.interp(yi, yp, z_x[:, j].astype(np.float64, copy=False))
    return z_xy


def _upsample_grid(z: np.ndarray, up_x: int = 6, up_y: int = 4) -> np.ndarray:
    if up_x <= 1 and up_y <= 1:
        return z
    try:
        from scipy.ndimage import zoom

        return zoom(z, (up_y, up_x), order=1)
    except ImportError:
        return _upsample_grid_numpy(z, up_x, up_y)


def _chart_3d_token_limit(outliers: dict[str, Any]) -> int:
    v = outliers.get("chart_3d_max_tokens", 24)
    try:
        return max(4, int(v))
    except (TypeError, ValueError):
        return 24


def _chart_token_trend_limit(outliers: dict[str, Any]) -> int:
    """Max number of token-trend PNGs (defaults to chart_3d_max_tokens cap)."""
    v = outliers.get("chart_token_trend_max_tokens")
    if v is None:
        return _chart_3d_token_limit(outliers)
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return _chart_3d_token_limit(outliers)


def _subsample_token_axis(
    tokens: list[str],
    indices: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    idx = indices.astype(int)
    tok = [tokens[i] for i in idx]
    return tok, idx


def _subsample_rows_list2d(
    rows: list[list[float]], idx: np.ndarray
) -> list[list[float]]:
    return [rows[int(i)] for i in idx]


def _subsample_cols_list2d(
    max_abs: list[list[float]], idx: np.ndarray
) -> list[list[float]]:
    out = []
    for row in max_abs:
        out.append([row[int(i)] for i in idx])
    return out


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

    num_layers = len(layer_names)
    if len(means) != num_layers or len(variances) != num_layers:
        return

    means_arr = np.asarray(means, dtype=np.float32)
    vars_arr = np.asarray(variances, dtype=np.float32)
    if means_arr.shape != (num_layers, token_count):
        return
    if vars_arr.shape != (num_layers, token_count):
        return

    trend_dir = output_dir / "token_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(num_layers, dtype=np.float32)
    xtick_labels = [_pretty_layer_name(n) for n in layer_names]

    max_plots = _chart_token_trend_limit(outliers)
    if token_count > max_plots:
        token_indices = np.linspace(
            0, token_count - 1, max_plots, dtype=np.int64
        )
    else:
        token_indices = np.arange(token_count, dtype=np.int64)

    fig_w = max(12.0, num_layers * 0.4)

    for token_idx in token_indices:
        ti = int(token_idx)
        y = means_arr[:, ti]
        vrow = vars_arr[:, ti]
        std = np.sqrt(np.maximum(vrow, 0.0))

        tok_label = None
        if isinstance(tokens, list) and ti < len(tokens):
            tok_label = tokens[ti]

        fig, ax = plt.subplots(figsize=(fig_w, 5))
        ax.plot(x, y, marker="o", linewidth=2.0, color="#3366cc")
        ax.fill_between(
            x, y - std, y + std, alpha=0.2, color="#6699ff", label="mean +/- std"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_xlabel("Layer name")
        ax.set_ylabel("Token mean activation (over hidden dims)")
        title = f"Token idx {ti}: mean activation across layers"
        if tok_label is not None:
            title = f"Token '{tok_label}': mean activation across layers"
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(trend_dir / f"token_{ti}.png", dpi=150)
        plt.close(fig)


def _plot_outlier_token_feature_3d(outliers: dict[str, Any], output_dir: Path) -> None:
    tokens = outliers.get("prompt_tokens", [])
    feat_dims = outliers.get("outlier_feature_dims", [])
    magnitudes = outliers.get("token_feature_magnitude", None)  # [T][F]
    if not tokens or not feat_dims or magnitudes is None:
        return

    token_count = len(tokens)
    f = len(feat_dims)
    if token_count == 0 or f == 0:
        return
    if len(magnitudes) != token_count or any(len(row) != f for row in magnitudes):
        return

    max_3d = _chart_3d_token_limit(outliers)
    if token_count > max_3d:
        idx = np.linspace(0, token_count - 1, max_3d, dtype=int)
        tokens, _ = _subsample_token_axis(tokens, idx)
        magnitudes = _subsample_rows_list2d(magnitudes, idx)
        token_count = len(tokens)

    mags = np.array(magnitudes, dtype=np.float32)  # [T, F]

    max_per_feat = mags.max(axis=0)  # [F]
    topk = min(3, f)
    top_feat_idx = np.argsort(-max_per_feat)[:topk]
    top_feat_idx = np.sort(top_feat_idx)

    feat_dims_sel = [feat_dims[int(i)] for i in top_feat_idx]
    mags = mags[:, top_feat_idx]  # [T, topk]
    f_plot = len(feat_dims_sel)

    mags_up = _upsample_grid(mags.T)  # [F_up, T_up]
    f_up, t_up = mags_up.shape

    xticks = np.linspace(0, t_up - 1, token_count)
    yticks = np.linspace(0, f_up - 1, f_plot)
    feat_labels = [str(d) for d in feat_dims_sel]

    fig = plt.figure(figsize=(max(12, token_count * 0.6), 7))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.arange(t_up), np.arange(f_up))
    ax.plot_wireframe(
        X, Y, mags_up, rstride=1, cstride=1, color="royalblue", linewidth=1.5
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(tokens, rotation=50, ha="right")
    ax.set_yticks(yticks)
    ax.set_yticklabels(feat_labels)
    ax.set_xlabel("Token", labelpad=20)
    ax.set_ylabel("Feature dim", labelpad=20)
    ax.set_zlabel("Magnitude (abs max over layers)", labelpad=20)
    ax.tick_params(axis="x", which="major")
    ax.tick_params(axis="y", which="major")
    ax.tick_params(axis="z", which="major")
    plt.setp(
        ax.get_xticklabels(),
        rotation=50,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")
    plt.title("Outlier feature magnitudes by token")
    plt.tight_layout()
    plt.savefig(output_dir / "outlier_token_feature_3d.png", dpi=200)
    plt.close()


def _plot_outlier_token_feature_3d_per_layer(
    outliers: dict[str, Any], output_dir: Path
) -> None:
    tokens = outliers.get("prompt_tokens", [])
    feat_dims = outliers.get("outlier_feature_dims", [])
    mags_by_layer = outliers.get(
        "token_feature_magnitude_by_layer", None
    )  # layer -> [T][F]
    if (
        not tokens
        or not feat_dims
        or not isinstance(mags_by_layer, dict)
        or not mags_by_layer
    ):
        return

    token_count = len(tokens)
    f = len(feat_dims)
    if token_count == 0 or f == 0:
        return

    per_layer_dir = output_dir / "outlier_token_feature_3d_per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)

    max_3d = _chart_3d_token_limit(outliers)
    idx = None
    tokens_plot = tokens
    if len(tokens) > max_3d:
        idx = np.linspace(0, len(tokens) - 1, max_3d, dtype=int)
        tokens_plot, _ = _subsample_token_axis(tokens, idx)

    xticks_template = None

    for layer_name, magnitudes in mags_by_layer.items():
        if len(magnitudes) != len(tokens) or any(len(row) != f for row in magnitudes):
            continue

        mags_rows = magnitudes
        if idx is not None:
            mags_rows = _subsample_rows_list2d(magnitudes, idx)
        t_plot = len(mags_rows)

        mags_tf = np.array(mags_rows, dtype=np.float32)  # [T, F]
        max_per_feat = mags_tf.max(axis=0)
        topk = min(3, f)
        top_feat_idx = np.argsort(-max_per_feat)[:topk]
        top_feat_idx = np.sort(top_feat_idx)
        feat_dims_top = [feat_dims[int(i)] for i in top_feat_idx]
        mags_tf = mags_tf[:, top_feat_idx]  # [T, topk]

        mags_up = _upsample_grid(mags_tf.T)  # [F_up, T_up]
        f_up, t_up = mags_up.shape
        if xticks_template is None or xticks_template.shape[0] != t_plot:
            xticks_template = np.linspace(0, t_up - 1, t_plot)

        pretty_layer = _pretty_layer_name(layer_name)
        fname_base = _sanitize_filename(pretty_layer)

        feat_labels = [str(d) for d in feat_dims_top]
        yticks = np.linspace(0, f_up - 1, len(feat_dims_top))

        fig = plt.figure(figsize=(max(12, t_plot * 0.6), 7))
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(t_up), np.arange(f_up))
        ax.plot_wireframe(
            X, Y, mags_up, rstride=1, cstride=1, color="royalblue", linewidth=1.5
        )
        ax.set_xlabel("Token", labelpad=20)
        ax.set_ylabel("Feature dim", labelpad=20)
        ax.set_zlabel("Magnitude (abs)", labelpad=20)
        ax.set_xticks(xticks_template)
        ax.set_xticklabels(tokens_plot, rotation=50, ha="right")
        ax.set_yticks(yticks)
        ax.set_yticklabels(feat_labels)
        ax.tick_params(axis="x", which="major")
        ax.tick_params(axis="y", which="major")
        ax.tick_params(axis="z", which="major")
        plt.setp(
            ax.get_xticklabels(),
            rotation=50,
            ha="right",
            va="center",
            rotation_mode="anchor",
        )
        plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")
        plt.title(f"{pretty_layer}: outlier feature magnitudes")
        plt.tight_layout()
        plt.savefig(per_layer_dir / f"{fname_base}.png", dpi=200)
        plt.close()


def _group_token_layer_trends_by_operator(
    raw_layer_names: list[str],
    max_abs: list[list[float]],
) -> dict[str, list[tuple[int, list[float]]]]:
    groups: dict[str, list[tuple[int, list[float]]]] = {}
    for layer_name, row in zip(raw_layer_names, max_abs, strict=False):
        block, operator = _parse_block_operator(layer_name)
        if block is None or operator is None:
            continue
        groups.setdefault(operator, []).append((block, list(row)))
    for operator in groups:
        groups[operator].sort(key=lambda item: item[0])
    return groups


def _plot_token_layer_3d(outliers: dict[str, Any], output_dir: Path) -> None:
    trends = outliers.get("token_layer_trends", None)
    if not isinstance(trends, dict):
        return

    raw_layer_names = trends.get("layer_names", [])
    max_abs = trends.get("max_abs", None)  # [L][T]
    tokens = trends.get("tokens", [])
    if not tokens or not raw_layer_names or max_abs is None:
        return

    token_count = len(tokens)
    if token_count == 0:
        return
    if len(max_abs) != len(raw_layer_names) or any(
        len(row) != token_count for row in max_abs
    ):
        return

    groups = _group_token_layer_trends_by_operator(raw_layer_names, max_abs)
    if not groups:
        return

    chart_dir = output_dir / "token_layer_maxabs_3d"
    chart_dir.mkdir(parents=True, exist_ok=True)

    max_3d = _chart_3d_token_limit(outliers)
    tokens_plot = tokens
    token_idx: np.ndarray | None = None
    if token_count > max_3d:
        token_idx = np.linspace(0, token_count - 1, max_3d, dtype=int)
        tokens_plot, _ = _subsample_token_axis(tokens, token_idx)
    t_plot = len(tokens_plot)

    thr = float(outliers.get("threshold", 6.0))

    for operator, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue

        block_ids = [int(b) for b, _ in rows]
        layer_rows = [row for _, row in rows]
        if token_idx is not None:
            layer_rows = [_subsample_cols_list2d([row], token_idx)[0] for row in layer_rows]

        mags = np.asarray(layer_rows, dtype=np.float32)  # [B, T]
        if mags.ndim != 2 or mags.shape[1] != t_plot:
            continue

        keep = np.where((mags >= thr).any(axis=1))[0]
        if keep.size == 0:
            keep = np.arange(mags.shape[0])
        mags = mags[keep, :]
        block_labels = [str(block_ids[int(i)]) for i in keep]
        block_count = len(block_labels)

        mags_up = _upsample_grid(mags, up_x=6, up_y=2)  # [B_up, T_up]
        b_up, t_up = mags_up.shape

        xticks = np.linspace(0, t_up - 1, t_plot)
        yticks = np.linspace(0, b_up - 1, block_count)

        fig = plt.figure(
            figsize=(max(12, t_plot * 0.6), max(7, block_count * 0.45))
        )
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(t_up), np.arange(b_up))
        ax.plot_wireframe(
            X, Y, mags_up, rstride=1, cstride=1, color="royalblue", linewidth=1.5
        )
        ax.set_xlabel("Token", labelpad=25)
        ax.set_ylabel("Transformer block")
        ax.set_zlabel("Max |activation| over hidden dims")
        ax.set_xticks(xticks)
        ax.set_xticklabels(tokens_plot, rotation=50, ha="right")
        ax.set_yticks(yticks)
        ax.set_yticklabels(block_labels)
        ax.tick_params(axis="x", which="major", pad=-4)
        ax.tick_params(axis="y", which="major", pad=-5)
        ax.tick_params(axis="z", which="major", pad=-1)
        plt.setp(
            ax.get_xticklabels(),
            rotation=50,
            ha="right",
            va="center",
            rotation_mode="anchor",
        )
        plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")
        ax.set_title(f"{operator}: per-token max |activation| across blocks")
        plt.tight_layout()
        fig.savefig(chart_dir / f"{_sanitize_filename(operator)}.png", dpi=200)
        plt.close(fig)


def _parse_block_operator(layer_name: str) -> tuple[int | None, str | None]:
    """Return (block_index, operator_suffix) for stacked transformer blocks."""
    parts = layer_name.split(".")
    if len(parts) >= 4 and parts[0] == "transformer" and parts[1] == "h":
        try:
            block = int(parts[2])
            operator = ".".join(parts[3:])
            return (block, operator) if operator else (block, None)
        except ValueError:
            pass
    for i, part in enumerate(parts):
        if part not in ("h", "layer", "layers"):
            continue
        if i + 1 >= len(parts):
            continue
        try:
            block = int(parts[i + 1])
        except ValueError:
            continue
        operator = ".".join(parts[i + 2 :])
        if operator:
            return block, operator
    return None, None


def _group_layers_by_operator_metric(
    layers: dict[str, Any],
    metric_key: str,
) -> dict[str, list[tuple[int, float]]]:
    groups: dict[str, list[tuple[int, float]]] = {}
    for layer_name, layer_stats in layers.items():
        if not isinstance(layer_stats, dict):
            continue
        val_raw = layer_stats.get(metric_key)
        if not isinstance(val_raw, (int, float)) or val_raw != val_raw:
            continue
        block, operator = _parse_block_operator(layer_name)
        if block is None or operator is None:
            continue
        groups.setdefault(operator, []).append((block, float(val_raw)))
    for operator in groups:
        groups[operator].sort(key=lambda row: row[0])
    return groups


def _group_layers_by_operator(
    layers: dict[str, Any],
) -> dict[str, list[tuple[int, str, dict[str, float]]]]:
    """operator -> [(block_idx, layer_name, activation_abs tops), ...]."""
    groups: dict[str, list[tuple[int, str, dict[str, float]]]] = {}
    for layer_name, layer_stats in layers.items():
        if not isinstance(layer_stats, dict):
            continue
        tops_raw = layer_stats.get("activation_abs")
        if not isinstance(tops_raw, dict) or not tops_raw:
            continue
        tops: dict[str, float] = {}
        for key in _OPERATOR_TOP_KEYS:
            val = tops_raw.get(key)
            if isinstance(val, (int, float)) and val == val:
                tops[key] = float(val)
        if not tops:
            continue
        block, operator = _parse_block_operator(layer_name)
        if block is None or operator is None:
            continue
        groups.setdefault(operator, []).append((block, layer_name, tops))
    for operator in groups:
        groups[operator].sort(key=lambda row: row[0])
    return groups


def _plot_operator_activation_tops(layers: dict[str, Any], output_dir: Path) -> None:
    groups = _group_layers_by_operator(layers)
    if not groups:
        return

    chart_dir = output_dir / "operator_activation_tops"
    chart_dir.mkdir(parents=True, exist_ok=True)

    palette = [
        "#3366cc",
        "#dc3912",
        "#ff9900",
        "#109618",
        "#990099",
        "#0099c6",
        "#dd4477",
        "#66aa00",
        "#b82e2e",
    ]

    for operator, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue

        block_ids = [int(r[0]) for r in rows]
        x = np.arange(len(block_ids), dtype=np.float32)

        fig, ax = plt.subplots(figsize=(max(10, len(block_ids) * 0.55), 5.5))
        for series_idx, key in enumerate(_OPERATOR_TOP_KEYS):
            y: list[float] = []
            for _, _, tops in rows:
                val = tops.get(key)
                y.append(float(val) if val is not None else float("nan"))
            if not any(v == v for v in y):
                continue
            color = palette[series_idx % len(palette)]
            label = _OPERATOR_TOP_LABELS.get(key, key)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in block_ids])
        ax.set_xlabel("Transformer block index")
        ax.set_ylabel("|activation| magnitude")
        ax.set_title(f"{operator}: |activation| tops across blocks")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(chart_dir / f"{_sanitize_filename(operator)}.png", dpi=150)
        plt.close(fig)


def _plot_operator_metric_aggregated(
    layers: dict[str, Any],
    output_dir: Path,
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    groups = _group_layers_by_operator_metric(layers, metric_key)
    if not groups:
        return

    operators = sorted(
        op for op, rows in groups.items() if len(rows) >= 2
    )
    if not operators:
        return

    palette = [
        "#3366cc",
        "#dc3912",
        "#ff9900",
        "#109618",
        "#990099",
        "#0099c6",
        "#dd4477",
        "#66aa00",
        "#b82e2e",
        "#316395",
        "#994499",
        "#22aa99",
    ]

    fig, ax = plt.subplots(figsize=(max(10, max(len(groups[op]) for op in operators) * 0.55), 6))
    for series_idx, operator in enumerate(operators):
        rows = groups[operator]
        block_ids = [int(b) for b, _ in rows]
        values = [float(v) for _, v in rows]
        x = np.asarray(block_ids, dtype=np.float32)
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            color=palette[series_idx % len(palette)],
            label=operator,
        )

    ax.set_xlabel("Transformer block index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def draw_activation_charts(
    stats: dict[str, Any],
    output_dir: str,
    *,
    draw_token_trends: bool = False,
) -> None:
    layers = stats.get("layers", {})
    if not layers:
        return
    outliers = stats.get("outliers", {})

    logger.info("[acta] creating charts in: %s", output_dir)
    rc = {
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with plt.rc_context(rc):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            if isinstance(outliers, dict):
                if draw_token_trends:
                    _plot_token_trends(outliers, out)
                _plot_outlier_token_feature_3d(outliers, out)
                _plot_outlier_token_feature_3d_per_layer(outliers, out)
                _plot_token_layer_3d(outliers, out)
            _plot_operator_activation_tops(layers, out)
            _plot_operator_metric_aggregated(
                layers,
                out,
                metric_key="max_kurtosis",
                title="Max channel kurtosis across transformer blocks",
                ylabel="Max kurtosis",
                filename="operator_max_kurtosis_across_blocks.png",
            )
            _plot_operator_metric_aggregated(
                layers,
                out,
                metric_key="max_median_ratio",
                title="Max-median ratio (MRR) across transformer blocks",
                ylabel="Max-median ratio",
                filename="operator_max_median_ratio_across_blocks.png",
            )


def build_charts_from_stats_file(
    stats_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    draw_token_trends: bool | None = None,
) -> Path:
    stats_file = Path(stats_path).expanduser().resolve()
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else stats_file.parent
    )
    if draw_token_trends is None:
        meta = stats.get("_acta")
        if isinstance(meta, dict):
            draw_token_trends = bool(meta.get("draw_token_trends", False))
        else:
            draw_token_trends = False
    draw_activation_charts(
        stats=stats,
        output_dir=out_dir.as_posix(),
        draw_token_trends=draw_token_trends,
    )
    return out_dir
