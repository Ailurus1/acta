from __future__ import annotations

import atexit
import copy
import csv
import fnmatch
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Union

import torch
from torch import nn

from .visualizer import draw_activation_charts

logger = logging.getLogger("acta.analyzer")


def _infer_acta_model_name(model: nn.Module) -> str:
    np_attr = getattr(model, "name_or_path", None)
    if isinstance(np_attr, str) and np_attr.strip():
        return np_attr.strip()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("name_or_path", "_name_or_path"):
            v = getattr(cfg, key, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
    default_cfg = getattr(model, "default_cfg", None)
    if isinstance(default_cfg, dict):
        tag = default_cfg.get("tag") or default_cfg.get("architecture")
        if isinstance(tag, str) and tag.strip():
            return tag.strip()
    return model.__class__.__name__


def _maybe_release_accelerator_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _to_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


_ACTIVATION_TOP_RANKS = (1, 2, 3, 10)
_ACTIVATION_TOP_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("top_p01", 0.01),
    ("top_p10", 0.10),
    ("top_p50", 0.50),
    ("top_p90", 0.90),
    ("top_p99", 0.99),
)
_MAX_ABS_SAMPLE_SIZE = 1_000_000


def _maybe_subsample_flat(
    abs_flat: torch.Tensor, max_size: int = _MAX_ABS_SAMPLE_SIZE
) -> torch.Tensor:
    x = abs_flat.reshape(-1).float()
    n = int(x.numel())
    if n <= max_size:
        return x
    idx = torch.randint(0, n, (max_size,))
    return x[idx]


def _compute_activation_abs_top_stats(abs_flat: torch.Tensor) -> dict[str, float]:
    x = abs_flat.reshape(-1).float()
    if x.numel() == 0:
        return {}
    sorted_desc = torch.sort(x, descending=True).values
    n = int(sorted_desc.numel())
    out: dict[str, float] = {}
    for k in _ACTIVATION_TOP_RANKS:
        out[f"top_{k}"] = (
            float(sorted_desc[k - 1].item()) if k <= n else float("nan")
        )
    qs = torch.tensor([p for _, p in _ACTIVATION_TOP_PERCENTILES], dtype=x.dtype)
    pct_vals = torch.quantile(x, qs)
    for i, (label, _) in enumerate(_ACTIVATION_TOP_PERCENTILES):
        out[label] = float(pct_vals[i].item())
    return out


def _global_kurtosis_from_flat(abs_flat: torch.Tensor) -> float | None:
    x = abs_flat.reshape(-1).float()
    if x.numel() < 2:
        return None
    mean = x.mean()
    centered = x - mean
    m2 = torch.mean(centered.pow(2))
    m4 = torch.mean(centered.pow(4))
    eps = torch.finfo(x.dtype).eps
    return float((m4 / (m2.pow(2) + eps)).item())


def _max_median_ratio_from_flat(abs_flat: torch.Tensor) -> float | None:
    x = abs_flat.reshape(-1).float()
    if x.numel() == 0:
        return None
    med = torch.median(x)
    eps = torch.finfo(x.dtype).eps
    return float((x.max() / med.clamp(min=eps)).item())


def _max_channel_kurtosis(kurtosis_values: list[Any]) -> float | None:
    vals: list[float] = []
    for v in kurtosis_values:
        if isinstance(v, (int, float)) and v == v:
            vals.append(float(v))
    if not vals:
        return None
    return float(max(vals))


def _compute_max_median_ratio(by_layer: dict[str, torch.Tensor]) -> float | None:
    tensors = [
        t
        for t in by_layer.values()
        if isinstance(t, torch.Tensor) and t.ndim == 2 and t.numel() > 0
    ]
    if not tensors:
        return None
    t_min = min(int(t.shape[0]) for t in tensors)
    h_min = min(int(t.shape[1]) for t in tensors)
    if t_min <= 0 or h_min <= 0:
        return None
    stack = torch.stack([t[:t_min, :h_min] for t in tensors], dim=0)
    max_abs = stack.max(dim=0).values
    med_abs = stack.median(dim=0).values
    eps = torch.finfo(stack.dtype).eps
    ratio = max_abs / med_abs.clamp(min=eps)
    return float(ratio.max().item())


def _channel_stats(values: torch.Tensor) -> Dict[str, Union[Dict, List[float]]]:
    if values.numel() == 0:
        return {
            "mean": [],
            "variance": [],
            "quantiles": {"q25": [], "q50": [], "q75": []},
            "kurtosis": [],
        }

    with torch.no_grad():
        mean = values.mean(dim=0)
        var = values.var(dim=0, unbiased=False)
        q = torch.quantile(
            values,
            q=torch.tensor([0.25, 0.5, 0.75], device=values.device, dtype=values.dtype),
            dim=0,
        )

        centered = values - mean
        m2 = torch.mean(centered.pow(2), dim=0)
        m4 = torch.mean(centered.pow(4), dim=0)
        eps = torch.finfo(values.dtype).eps
        kurtosis = m4 / (m2.pow(2) + eps)

        out_mean = mean.cpu().tolist()
        out_var = var.cpu().tolist()
        out_q25 = q[0].cpu().tolist()
        out_q50 = q[1].cpu().tolist()
        out_q75 = q[2].cpu().tolist()
        out_kurt = kurtosis.cpu().tolist()

        del mean, var, q, centered, m2, m4, kurtosis

    return {
        "mean": out_mean,
        "variance": out_var,
        "quantiles": {"q25": out_q25, "q50": out_q50, "q75": out_q75},
        "kurtosis": out_kurt,
    }


def _channel_stats_from_running_moments(
    *,
    count: int,
    sum_: torch.Tensor,
    sumsq: torch.Tensor,
    sum3: torch.Tensor | None = None,
    sum4: torch.Tensor | None = None,
) -> Dict[str, Union[Dict, List[float]]]:
    if count <= 0 or sum_.numel() == 0:
        return {
            "mean": [],
            "variance": [],
            "quantiles": {"q25": [], "q50": [], "q75": []},
            "kurtosis": [],
        }
    c = float(count)
    mean = sum_ / c
    var = torch.clamp((sumsq / c) - mean.pow(2), min=0.0)
    n_ch = int(mean.numel())
    empty = [None] * n_ch
    if sum3 is not None and sum4 is not None:
        ex2 = sumsq / c
        ex3 = sum3 / c
        ex4 = sum4 / c
        eps = torch.finfo(mean.dtype).eps
        m4_central = (
            ex4 - 4.0 * mean * ex3 + 6.0 * mean.pow(2) * ex2 - 3.0 * mean.pow(4)
        )
        kurtosis = m4_central / (var.pow(2) + eps)
        out_kurt = kurtosis.cpu().tolist()
    else:
        out_kurt = empty
    return {
        "mean": mean.cpu().tolist(),
        "variance": var.cpu().tolist(),
        "quantiles": {"q25": empty, "q50": empty, "q75": empty},
        "kurtosis": out_kurt,
    }


def _prepare_per_channel(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """
    Convert tensor to shape [N, C] where C is channel/features dimension.

    1D: single channel
    2D: treat last dim as channels (e.g., [batch, hidden])
    >=3D:
      - conv-like modules: treat dim=1 as channels ([N, C, ...])
      - other modules (e.g., transformer blocks): treat last dim as channels
    """
    x = x.detach().float()
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x.reshape(-1, x.shape[-1])

    is_conv_like = isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    )
    if is_conv_like:
        x = x.movedim(1, -1)
    return x.reshape(-1, x.shape[-1])


def _token_outlier_fraction(x: torch.Tensor, threshold: float) -> torch.Tensor | None:
    """
    Return fraction of hidden dims with |activation| >= threshold per token.

    Expects transformer-like activation tensor:
      - [B, S, H] -> returns [B, S]
      - [S, B, H] -> returns [B, S]

    Other shapes return None.
    """
    if not (x.is_floating_point() and x.ndim == 3):
        return None

    frac = (x.detach().abs() >= threshold).to(torch.float32).mean(dim=-1)
    return frac


def _format_outlier_table_decoded(
    tokens: list[str],
    token_ids: list[int],
    soft_outliers: list[bool],
    hard_outliers: list[bool],
    interquantile_outliers: list[bool],
    layers: list[str | None],
    channel_dims: list[int | None],
    soft_layers: list[str | None] | None = None,
    soft_dims: list[int | None] | None = None,
    hard_layers: list[str | None] | None = None,
    hard_dims: list[int | None] | None = None,
    iqr_layers: list[str | None] | None = None,
    iqr_dims: list[int | None] | None = None,
    outliers_detected: list[bool] | None = None,
    *,
    stdout_only: bool = True,
) -> str:
    def _token_display(s: str) -> str:
        s = s.replace("\n", "\\n")
        return s if len(s) <= 20 else s[:19] + "…"

    def _cell_layer(v: str | None) -> str:
        return "" if v is None else str(v).replace("\n", "\\n")

    def _cell_dim(v: int | None) -> str:
        return "" if v is None else str(v)

    soft_layer_col = _attribution_layer_key(_SOFT_FLAG)
    soft_dim_col = _attribution_dim_key(_SOFT_FLAG)
    hard_layer_col = _attribution_layer_key(_HARD_FLAG)
    hard_dim_col = _attribution_dim_key(_HARD_FLAG)
    iqr_layer_col = _attribution_layer_key(_IQR_FLAG)
    iqr_dim_col = _attribution_dim_key(_IQR_FLAG)

    if stdout_only:
        column_names = [
            "idx",
            "token",
            "token_id",
            "layer",
            "channel_dim",
            _OUTLIERS_DETECTED,
        ]
    else:
        column_names = [
            "idx",
            "token",
            "token_id",
            "layer",
            "channel_dim",
            _SOFT_FLAG,
            soft_layer_col,
            soft_dim_col,
            _HARD_FLAG,
            hard_layer_col,
            hard_dim_col,
            _IQR_FLAG,
            iqr_layer_col,
            iqr_dim_col,
            _OUTLIERS_DETECTED,
        ]
    column_caps = {
        "layer": 28,
        soft_layer_col: 28,
        hard_layer_col: 28,
        iqr_layer_col: 28,
        "token": 16,
    }

    def _fit_cell(cell: str, width: int) -> str:
        if len(cell) <= width:
            return cell
        if width <= 1:
            return cell[:width]
        return cell[: width - 1] + "…"

    def _fmt_row(cells: list[str], widths: list[int]) -> str:
        parts: list[str] = []
        for i, cell in enumerate(cells):
            w = widths[i]
            fitted = _fit_cell(cell, w)
            if i == 0:
                parts.append(f"{fitted:>{w}}")
            else:
                parts.append(f"{fitted:<{w}}")
        return " ".join(parts)

    data_rows: list[list[str]] = []
    soft_layers = soft_layers or []
    soft_dims = soft_dims or []
    hard_layers = hard_layers or []
    hard_dims = hard_dims or []
    iqr_layers = iqr_layers or []
    iqr_dims = iqr_dims or []
    outliers_detected = outliers_detected or []

    for i, (tok, tid, layer, dim, soft_o, hard_o, iqr_o) in enumerate(
        zip(
            tokens,
            token_ids,
            layers,
            channel_dims,
            soft_outliers,
            hard_outliers,
            interquantile_outliers,
            strict=False,
        )
    ):
        sl = soft_layers[i] if i < len(soft_layers) else None
        sd = soft_dims[i] if i < len(soft_dims) else None
        hl = hard_layers[i] if i < len(hard_layers) else None
        hd = hard_dims[i] if i < len(hard_dims) else None
        il = iqr_layers[i] if i < len(iqr_layers) else None
        ide = iqr_dims[i] if i < len(iqr_dims) else None
        od = outliers_detected[i] if i < len(outliers_detected) else (
            bool(soft_o or hard_o or iqr_o)
        )
        if stdout_only:
            data_rows.append(
                [
                    str(i),
                    _token_display(tok),
                    str(tid),
                    _cell_layer(layer),
                    _cell_dim(dim),
                    str(od),
                ]
            )
        else:
            data_rows.append(
                [
                    str(i),
                    _token_display(tok),
                    str(tid),
                    _cell_layer(layer),
                    _cell_dim(dim),
                    str(soft_o),
                    _cell_layer(sl),
                    _cell_dim(sd),
                    str(hard_o),
                    _cell_layer(hl),
                    _cell_dim(hd),
                    str(iqr_o),
                    _cell_layer(il),
                    _cell_dim(ide),
                    str(od),
                ]
            )

    widths = [len(name) for name in column_names]
    for row in data_rows:
        for i, cell in enumerate(row):
            cap = column_caps.get(column_names[i])
            cell_len = len(cell)
            if cap is not None:
                cell_len = min(cell_len, cap)
            widths[i] = max(widths[i], cell_len)
    for i, name in enumerate(column_names):
        cap = column_caps.get(name)
        if cap is not None:
            widths[i] = min(widths[i], cap)

    header = _fmt_row(column_names, widths)
    lines = [header, "-" * len(header)]
    for row in data_rows:
        lines.append(_fmt_row(row, widths))
    return "\n".join(lines)


_SOFT_FLAG = "llm.int8() outliers soft difinition"
_HARD_FLAG = "llm.int8() outliers hard definition"
_IQR_FLAG = "interquantile_outliers"
_OUTLIERS_DETECTED = "outliers detected"


def _attribution_layer_key(flag_key: str) -> str:
    return f"{flag_key}_layer"


def _attribution_dim_key(flag_key: str) -> str:
    return f"{flag_key}_channel_dim"


def _sync_outlier_attribution_columns(payload: dict[str, Any]) -> None:
    """Clear per-modality layer/dim when that modality's flag is false; mirror hard columns to legacy keys."""

    def _flag_list(fk: str, alt: str | None) -> list[Any] | None:
        v = payload.get(fk)
        if isinstance(v, list):
            return v
        if alt is not None:
            v2 = payload.get(alt)
            if isinstance(v2, list):
                return v2
        return None

    hf = _flag_list(_HARD_FLAG, "outliers")
    if isinstance(hf, list):
        n0 = len(hf)
        hk = _attribution_layer_key(_HARD_FLAG)
        dk = _attribution_dim_key(_HARD_FLAG)
        hl = list(payload.get(hk, []))
        hd = list(payload.get(dk, []))
        pt_l = list(payload.get("per_token_layer", []))
        pt_d = list(payload.get("per_token_channel_dim", []))
        while len(hl) < n0:
            i = len(hl)
            hl.append(pt_l[i] if i < len(pt_l) else None)
        while len(hd) < n0:
            i = len(hd)
            hd.append(pt_d[i] if i < len(pt_d) else None)
        payload[hk] = hl[:n0]
        payload[dk] = hd[:n0]

    modality_pairs: list[tuple[str, str | None]] = [
        (_SOFT_FLAG, None),
        (_HARD_FLAG, "outliers"),
        (_IQR_FLAG, None),
    ]
    for fk, alt in modality_pairs:
        flags = _flag_list(fk, alt)
        if not isinstance(flags, list):
            continue
        n = len(flags)
        lk = _attribution_layer_key(fk)
        dk = _attribution_dim_key(fk)
        layers = list(payload.get(lk, []))
        dims = list(payload.get(dk, []))
        while len(layers) < n:
            layers.append(None)
        while len(dims) < n:
            dims.append(None)
        del layers[n:]
        del dims[n:]
        for i in range(n):
            if not bool(flags[i]):
                layers[i] = None
                dims[i] = None
        payload[lk] = layers
        payload[dk] = dims

    hl = payload.get(_attribution_layer_key(_HARD_FLAG))
    hd = payload.get(_attribution_dim_key(_HARD_FLAG))
    if isinstance(hl, list) and isinstance(hd, list):
        payload["per_token_layer"] = list(hl)
        payload["per_token_channel_dim"] = list(hd)


def _outliers_detected_row(soft: bool, hard: bool, iqr: bool) -> bool:
    return bool(soft or hard or iqr)


def _compute_outliers_detected_list(
    soft: list[bool],
    hard: list[bool],
    iqr: list[bool],
) -> list[bool]:
    n = max(len(soft), len(hard), len(iqr))
    out: list[bool] = []
    for i in range(n):
        s = bool(soft[i]) if i < len(soft) else False
        h = bool(hard[i]) if i < len(hard) else False
        iq = bool(iqr[i]) if i < len(iqr) else False
        out.append(_outliers_detected_row(s, h, iq))
    return out


def _best_layer_for_hidden_dim(
    by_layer: dict[str, torch.Tensor],
    ti: int,
    h: int,
) -> str | None:
    best_ln: str | None = None
    best_v = -1.0
    for ln, ten in by_layer.items():
        if ten.ndim != 2 or ti >= ten.shape[0] or h >= ten.shape[1]:
            continue
        v = float(ten[ti, h].item())
        if v > best_v:
            best_v = v
            best_ln = ln
    return best_ln


def _format_outlier_table_from_payload(o: dict[str, Any]) -> str:
    tokens = [str(v) for v in o.get("prompt_tokens", [])]
    token_ids_raw = o.get("prompt_token_ids", [])
    if isinstance(token_ids_raw, list):
        token_ids = [
            int(v) if isinstance(v, int) else int(i)
            for i, v in enumerate(token_ids_raw)
        ]
    else:
        token_ids = []
    if len(token_ids) != len(tokens):
        token_ids = list(range(len(tokens)))

    soft = [bool(v) for v in o.get(_SOFT_FLAG, [])]
    hard_src = o.get(_HARD_FLAG)
    if not isinstance(hard_src, list):
        hard_src = o.get("outliers", [])
    hard = [bool(v) for v in hard_src] if isinstance(hard_src, list) else []
    iqr = [bool(v) for v in o.get(_IQR_FLAG, [])]
    layers = [v if v not in ("", None) else None for v in o.get("per_token_layer", [])]
    dims_raw = o.get("per_token_channel_dim", [])
    channel_dims: list[int | None] = []
    for v in dims_raw:
        if v is None or v == "":
            channel_dims.append(None)
        else:
            try:
                channel_dims.append(int(v))
            except (TypeError, ValueError):
                channel_dims.append(None)

    od = o.get(_OUTLIERS_DETECTED)
    if not isinstance(od, list):
        od = _compute_outliers_detected_list(soft, hard, iqr)

    def _col(key: str) -> list[Any]:
        v = o.get(key)
        return list(v) if isinstance(v, list) else []

    return _format_outlier_table_decoded(
        tokens=tokens,
        token_ids=token_ids,
        soft_outliers=soft,
        hard_outliers=hard,
        interquantile_outliers=iqr,
        layers=[str(x) if x is not None else None for x in layers],
        channel_dims=channel_dims,
        soft_layers=_col(_attribution_layer_key(_SOFT_FLAG)),
        soft_dims=_col(_attribution_dim_key(_SOFT_FLAG)),
        hard_layers=_col(_attribution_layer_key(_HARD_FLAG)),
        hard_dims=_col(_attribution_dim_key(_HARD_FLAG)),
        iqr_layers=_col(_attribution_layer_key(_IQR_FLAG)),
        iqr_dims=_col(_attribution_dim_key(_IQR_FLAG)),
        outliers_detected=[bool(x) for x in od] if od else None,
        stdout_only=True,
    )


def decode_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            pass
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            tok = tokenizer.convert_ids_to_tokens([token_id])[0]
            return tok if tok is not None else str(token_id)
        except Exception:
            pass
    return str(token_id)


class _AnalyzerModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dump_stats_path: str | None,
        target_layers: str | list[str] | None = None,
        draw_charts: bool = False,
        draw_token_trends: bool = False,
        outlier_threshold: float = 6.0,
        hard_layers_frac: float = 0.25,
        hard_seqdim_frac: float = 0.06,
        verbose: bool = False,
        tokenizer: Any | None = None,
        vit_reg_patch_labels: bool = False,
        asr_chunk_labels: bool = False,
        chart_3d_max_tokens: int = 24,
        finalize_on_exit: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self._dump_stats_path_spec = dump_stats_path
        self._session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dump_stats_root, self.output_run_dir, self.dump_stats_path = (
            self._resolve_versioned_dump_paths()
        )
        self.draw_charts = draw_charts
        self.draw_token_trends = bool(draw_token_trends)
        self.outlier_threshold = float(outlier_threshold)
        self.hard_layers_frac = float(hard_layers_frac)
        self.hard_seqdim_frac = float(hard_seqdim_frac)
        self.verbose = bool(verbose)
        if self.verbose and not logging.getLogger("acta").handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            )
        self.tokenizer = tokenizer
        self.vit_reg_patch_labels = bool(vit_reg_patch_labels)
        self.asr_chunk_labels = bool(asr_chunk_labels)
        self.chart_3d_max_tokens = int(chart_3d_max_tokens)
        self._finalize_on_exit_enabled = bool(finalize_on_exit)
        tl_raw = target_layers
        if isinstance(tl_raw, str):
            tl_parts = [p.strip() for p in tl_raw.split(",") if p.strip()]
            target_layers = tl_parts if tl_parts else None
        self.target_layers = target_layers
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._layer_running_stats: dict[str, dict[str, torch.Tensor | int]] = {}
        self._in_generate: bool = False
        self._collect_layer_channel_stats: bool = True
        self._collect_outlier_payload: bool = True
        self._collect_token_trends: bool = True
        self._collect_feature_level: bool = True
        self._run_token_frac_by_layer: dict[str, list[torch.Tensor]] = {}
        self._run_token_mean_by_layer: dict[str, list[torch.Tensor]] = {}
        self._run_token_var_by_layer: dict[str, list[torch.Tensor]] = {}
        self._last_prompt_token_ids: list[int] | None = None
        self._prompt_len: int | None = None
        self._run_feature_token_count_by_layer: dict[str, int] = {}
        self._run_feature_counts_by_layer: dict[str, torch.Tensor] = {}
        self._run_max_abs_token_feature_by_layer: dict[
            str, torch.Tensor
        ] = {}  # layer -> [T_prompt, H]
        self._run_max_abs_token_feature: torch.Tensor | None = None  # [T_prompt, H]
        self._run_hidden_dim: int | None = None
        self._run_argmax_layer_per_token_feature: torch.Tensor | None = (
            None  # [T_prompt, H] int32
        )
        self._run_layer_order: list[str] = []
        self._run_layer_to_index: dict[str, int] = {}
        self._run_token_maxabs_by_layer: dict[str, list[torch.Tensor]] = {}
        self._run_abs_chunks: list[torch.Tensor] = []
        self._layer_abs_chunks: dict[str, list[torch.Tensor]] = {}
        self._outlier_table_printed: bool = False
        self._last_generate_outliers: dict[str, Any] | None = None
        self._aggregate_generate_outliers: dict[str, Any] | None = None
        self._final_dump_done: bool = False
        self._register_hooks()
        if self._finalize_on_exit_enabled:
            atexit.register(self._finalize_on_exit)

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info("[acta] %s", message)

    def _print_outlier_table(self, *, force: bool = False) -> None:
        agg = self._aggregate_generate_outliers
        if not isinstance(agg, dict):
            return
        tbl = agg.get("table")
        if not isinstance(tbl, str) or not tbl.strip():
            return
        if self._outlier_table_printed and not force:
            return
        if not force and not self.verbose:
            return
        print(tbl, flush=True)
        self._outlier_table_printed = True

    def _resolve_versioned_dump_paths(self) -> tuple[Path, Path, str]:
        """
        Returns (dump_root, run_dir, json_path) with layout::

            <dump_root>/<timestamp>/

        where ``dump_root`` is the user-facing output root (``dump_stats_path``
        when it names a directory).
        """
        spec = self._dump_stats_path_spec
        if spec is None or str(spec).strip() == "":
            dump_root = Path(".acta_dump_results")
            json_name = "stats.json"
        else:
            p = Path(spec)
            if p.suffix.lower() == ".json":
                dump_root = p.parent
                if dump_root == Path(""):
                    dump_root = Path(".")
                json_name = p.name
            else:
                dump_root = Path(spec)
                json_name = "stats.json"

        dump_root = dump_root.resolve()
        run_dir = (dump_root / self._session_timestamp).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        json_path = run_dir / json_name
        return dump_root, run_dir, str(json_path)

    def _write_acta_results_csv(self) -> None:
        path = self.output_run_dir / "acta_results.csv"
        soft_lk = _attribution_layer_key(_SOFT_FLAG)
        soft_dk = _attribution_dim_key(_SOFT_FLAG)
        hard_lk = _attribution_layer_key(_HARD_FLAG)
        hard_dk = _attribution_dim_key(_HARD_FLAG)
        iqr_lk = _attribution_layer_key(_IQR_FLAG)
        iqr_dk = _attribution_dim_key(_IQR_FLAG)
        fieldnames = [
            "idx",
            "token",
            "token_id",
            "layer",
            "channel_dim",
            _SOFT_FLAG,
            soft_lk,
            soft_dk,
            _HARD_FLAG,
            hard_lk,
            hard_dk,
            _IQR_FLAG,
            iqr_lk,
            iqr_dk,
            _OUTLIERS_DETECTED,
        ]
        rows: list[dict[str, Any]] = []
        if self._aggregate_generate_outliers is not None:
            agg = self._aggregate_generate_outliers
            tokens = agg.get("prompt_tokens", [])
            if not isinstance(tokens, list):
                tokens = []
            tids = agg.get("prompt_token_ids", [])
            if not isinstance(tids, list):
                tids = []
            soft_flags = agg.get(_SOFT_FLAG, [])
            if not isinstance(soft_flags, list):
                soft_flags = []
            hard_flags = agg.get(_HARD_FLAG)
            if not isinstance(hard_flags, list):
                hard_flags = agg.get("outliers", [])
            if not isinstance(hard_flags, list):
                hard_flags = []
            iqr_flags = agg.get(_IQR_FLAG, [])
            if not isinstance(iqr_flags, list):
                iqr_flags = []
            layers = agg.get("per_token_layer", [])
            if not isinstance(layers, list):
                layers = []
            dims = agg.get("per_token_channel_dim", [])
            if not isinstance(dims, list):
                dims = []
            soft_layers = agg.get(soft_lk, [])
            if not isinstance(soft_layers, list):
                soft_layers = []
            soft_dims = agg.get(soft_dk, [])
            if not isinstance(soft_dims, list):
                soft_dims = []
            hard_layers = agg.get(hard_lk, [])
            if not isinstance(hard_layers, list):
                hard_layers = []
            hard_dims = agg.get(hard_dk, [])
            if not isinstance(hard_dims, list):
                hard_dims = []
            iqr_layers = agg.get(iqr_lk, [])
            if not isinstance(iqr_layers, list):
                iqr_layers = []
            iqr_dims = agg.get(iqr_dk, [])
            if not isinstance(iqr_dims, list):
                iqr_dims = []
            od_list = agg.get(_OUTLIERS_DETECTED, [])
            if not isinstance(od_list, list):
                od_list = []

            def _b(lst: list[Any], i: int, default: Any = False) -> Any:
                return lst[i] if i < len(lst) else default

            def _layer_cell(lst: list[Any], i: int) -> str:
                v = _b(lst, i, None)
                return "" if v in (None, "") else str(v)

            def _dim_cell(lst: list[Any], i: int) -> str | int:
                v = _b(lst, i, None)
                return "" if v is None else v

            for i, tok in enumerate(tokens):
                s = bool(_b(soft_flags, i, False))
                h = bool(_b(hard_flags, i, False))
                iq = bool(_b(iqr_flags, i, False))
                od = (
                    bool(_b(od_list, i, False))
                    if isinstance(od_list, list)
                    else _outliers_detected_row(s, h, iq)
                )
                rows.append(
                    {
                        "idx": i,
                        "token": tok,
                        "token_id": tids[i] if i < len(tids) else "",
                        "layer": _layer_cell(layers, i),
                        "channel_dim": _dim_cell(dims, i),
                        _SOFT_FLAG: s,
                        soft_lk: _layer_cell(soft_layers, i),
                        soft_dk: _dim_cell(soft_dims, i),
                        _HARD_FLAG: h,
                        hard_lk: _layer_cell(hard_layers, i),
                        hard_dk: _dim_cell(hard_dims, i),
                        _IQR_FLAG: iq,
                        iqr_lk: _layer_cell(iqr_layers, i),
                        iqr_dk: _dim_cell(iqr_dims, i),
                        _OUTLIERS_DETECTED: od,
                    }
                )
        self.dump_stats_root.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _reset_run_sequence_buffers(self) -> None:
        self._run_token_frac_by_layer.clear()
        self._run_token_mean_by_layer.clear()
        self._run_token_var_by_layer.clear()
        self._last_generate_outliers = None
        self._last_prompt_token_ids = None
        self._prompt_len = None
        self._run_feature_token_count_by_layer.clear()
        self._run_feature_counts_by_layer.clear()
        self._run_max_abs_token_feature_by_layer.clear()
        self._run_max_abs_token_feature = None
        self._run_hidden_dim = None
        self._run_argmax_layer_per_token_feature = None
        self._run_layer_order.clear()
        self._run_layer_to_index.clear()
        self._run_token_maxabs_by_layer.clear()
        self._run_abs_chunks.clear()
        self._layer_abs_chunks.clear()

    def _current_run_batch_size(self) -> int:
        for chunks in self._run_token_frac_by_layer.values():
            if chunks:
                return int(chunks[0].shape[0])
        return 1

    def _maybe_set_prompt_len_from_tensor(self, tensor: torch.Tensor) -> None:
        if self._prompt_len is None and tensor.ndim == 3:
            self._prompt_len = int(tensor.shape[1])
            self._log(f"captured prompt length from hook: {self._prompt_len}")

    def _default_position_tokens(self, n: int) -> list[str]:
        if n <= 0:
            return []
        if n == 1:
            return ["cls"]
        return ["cls"] + [f"pos_{i}" for i in range(1, n)]

    def _sequence_position_labels(self, n: int) -> list[str]:
        """Labels for sequence positions when tokenizer is None (e.g. ViT patches, ASR chunks)."""
        if n <= 0:
            return []
        if self.asr_chunk_labels:
            return [f"chunk_{i}" for i in range(n)]
        if self.vit_reg_patch_labels:
            return [f"reg_{i}" for i in range(n)]
        return self._default_position_tokens(n)

    @property
    def device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return getattr(self.model, "device")
        return next(self.model.parameters()).device

    def _layer_is_targeted(self, layer_name: str, module: nn.Module) -> bool:
        if self.target_layers is None:
            return True

        patterns = (
            self.target_layers
            if isinstance(self.target_layers, list)
            else [self.target_layers]
        )
        class_name = module.__class__.__name__
        for pattern in patterns:
            if fnmatch.fnmatchcase(layer_name, pattern) or fnmatch.fnmatchcase(
                class_name, pattern
            ):
                return True

            try:
                if re.search(pattern, layer_name) or re.search(pattern, class_name):
                    return True
            except re.error:
                continue

        return False

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name == "":
                continue

            is_leaf = not any(True for _ in module.children())
            if self.target_layers is None:
                if not is_leaf:
                    continue
            else:
                if not self._layer_is_targeted(name, module):
                    continue

            def _hook_fn(
                mod: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                layer_name: str = name,
            ) -> None:
                tensor = _to_tensor(output)
                if tensor is None:
                    return
                if not tensor.is_floating_point():
                    return

                t_cpu: torch.Tensor | None = None
                if self._in_generate:
                    self._maybe_set_prompt_len_from_tensor(tensor)
                    if self._collect_outlier_payload:
                        t_cpu = tensor.detach().float().cpu()
                        frac = _token_outlier_fraction(
                            t_cpu, threshold=self.outlier_threshold
                        )
                        if frac is not None:
                            self._run_token_frac_by_layer.setdefault(
                                layer_name, []
                            ).append(frac)

                            if self._collect_token_trends:
                                token_mean = t_cpu.mean(dim=-1)  # [B, S]
                                token_var = t_cpu.var(dim=-1, unbiased=False)  # [B, S]
                                self._run_token_mean_by_layer.setdefault(
                                    layer_name, []
                                ).append(token_mean)
                                self._run_token_var_by_layer.setdefault(
                                    layer_name, []
                                ).append(token_var)

                            if (
                                self._collect_feature_level
                                and tensor.ndim == 3
                                and self._prompt_len is not None
                            ):
                                b, s, h = tensor.shape
                                t = min(int(self._prompt_len), int(s))
                                if t > 0:
                                    abs_act = t_cpu[:, :t, :].abs()  # [B, T, H]

                                    if self._collect_token_trends:
                                        tok_max = (
                                            abs_act.max(dim=-1).values.max(dim=0).values
                                        )  # [T]
                                        self._run_token_maxabs_by_layer.setdefault(
                                            layer_name, []
                                        ).append(tok_max)

                                    abs_cpu = abs_act.max(dim=0).values  # [T, H]
                                    prev_layer = self._run_max_abs_token_feature_by_layer.get(
                                        layer_name
                                    )
                                    if (
                                        prev_layer is None
                                        or prev_layer.shape != abs_cpu.shape
                                    ):
                                        self._run_max_abs_token_feature_by_layer[
                                            layer_name
                                        ] = abs_cpu
                                    else:
                                        self._run_max_abs_token_feature_by_layer[
                                            layer_name
                                        ] = torch.maximum(prev_layer, abs_cpu)

                                    if (
                                        self._run_max_abs_token_feature is None
                                        or self._run_hidden_dim != int(h)
                                    ):
                                        self._run_hidden_dim = int(h)
                                        self._run_max_abs_token_feature = abs_cpu
                                        layer_idx = self._run_layer_to_index.setdefault(
                                            layer_name, len(self._run_layer_order)
                                        )
                                        if layer_idx == len(self._run_layer_order):
                                            self._run_layer_order.append(layer_name)
                                        self._run_argmax_layer_per_token_feature = (
                                            torch.full(
                                                (t, int(h)),
                                                int(layer_idx),
                                                dtype=torch.int32,
                                            )
                                        )
                                    else:
                                        layer_idx = self._run_layer_to_index.setdefault(
                                            layer_name, len(self._run_layer_order)
                                        )
                                        if layer_idx == len(self._run_layer_order):
                                            self._run_layer_order.append(layer_name)
                                        prev_max = self._run_max_abs_token_feature
                                        updated_max = torch.maximum(prev_max, abs_cpu)
                                        self._run_max_abs_token_feature = updated_max
                                        if (
                                            self._run_argmax_layer_per_token_feature
                                            is not None
                                        ):
                                            upd = abs_cpu > prev_max
                                            self._run_argmax_layer_per_token_feature[
                                                upd
                                            ] = int(layer_idx)

                                    mask = abs_act >= float(self.outlier_threshold)
                                    self._run_feature_token_count_by_layer[layer_name] = (
                                        self._run_feature_token_count_by_layer.get(
                                            layer_name, 0
                                        )
                                        + (t * int(b))
                                    )
                                    feat_counts = mask.sum(dim=(0, 1))
                                    if layer_name not in self._run_feature_counts_by_layer:
                                        self._run_feature_counts_by_layer[layer_name] = (
                                            feat_counts
                                        )
                                    else:
                                        self._run_feature_counts_by_layer[layer_name] += (
                                            feat_counts
                                        )

                                    del abs_act, mask

                if self._collect_layer_channel_stats:
                    cpu_flat_in = (
                        t_cpu if t_cpu is not None else tensor.detach().float().cpu()
                    )
                    flattened = _prepare_per_channel(cpu_flat_in, mod)
                    n = int(flattened.shape[0])
                    if n > 0:
                        abs_sample = _maybe_subsample_flat(
                            flattened.abs().reshape(-1)
                        ).cpu()
                        self._run_abs_chunks.append(abs_sample)
                        self._layer_abs_chunks.setdefault(layer_name, []).append(
                            abs_sample
                        )
                        layer_state = self._layer_running_stats.get(layer_name)
                        sum_chunk = flattened.sum(dim=0)
                        sumsq_chunk = (flattened * flattened).sum(dim=0)
                        sum3_chunk = flattened.pow(3).sum(dim=0)
                        sum4_chunk = flattened.pow(4).sum(dim=0)
                        if layer_state is None:
                            self._layer_running_stats[layer_name] = {
                                "count": n,
                                "sum": sum_chunk,
                                "sumsq": sumsq_chunk,
                                "sum3": sum3_chunk,
                                "sum4": sum4_chunk,
                            }
                        else:
                            layer_state["count"] = int(layer_state["count"]) + n
                            layer_state["sum"] = layer_state["sum"] + sum_chunk
                            layer_state["sumsq"] = layer_state["sumsq"] + sumsq_chunk
                            layer_state["sum3"] = layer_state["sum3"] + sum3_chunk
                            layer_state["sum4"] = layer_state["sum4"] + sum4_chunk

            self._hooks.append(module.register_forward_hook(_hook_fn))
            self._log(f"hook registered for layer: {name}")

    def reset_stats(self) -> None:
        self._layer_running_stats.clear()

    def unregister_hooks(self) -> None:
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks.clear()

    def _build_stats(self) -> dict[str, Any]:
        self._log(
            f"aggregating activation statistics from {len(self._layer_running_stats)} layers"
        )
        layers: dict[str, Any] = {}
        for layer_name, state in self._layer_running_stats.items():
            count = int(state.get("count", 0))
            sum_ = state.get("sum")
            sumsq = state.get("sumsq")
            sum3 = state.get("sum3")
            sum4 = state.get("sum4")
            if count <= 0 or not isinstance(sum_, torch.Tensor) or not isinstance(
                sumsq, torch.Tensor
            ):
                continue
            layer_stats = _channel_stats_from_running_moments(
                count=count,
                sum_=sum_,
                sumsq=sumsq,
                sum3=sum3 if isinstance(sum3, torch.Tensor) else None,
                sum4=sum4 if isinstance(sum4, torch.Tensor) else None,
            )
            layer_stats["num_observations"] = count
            layer_chunks = self._layer_abs_chunks.get(layer_name, [])
            if layer_chunks:
                layer_cat = torch.cat(layer_chunks)
                layer_stats["activation_abs"] = _compute_activation_abs_top_stats(
                    layer_cat
                )
                layer_mrr = _max_median_ratio_from_flat(layer_cat)
                if layer_mrr is not None:
                    layer_stats["max_median_ratio"] = layer_mrr
            kurt_max = _max_channel_kurtosis(layer_stats.get("kurtosis", []))
            if kurt_max is not None:
                layer_stats["max_kurtosis"] = kurt_max
            layers[layer_name] = layer_stats
        return {"layers": layers}

    def _build_activation_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {}

        if self._run_abs_chunks:
            flat = torch.cat(self._run_abs_chunks)
            tops = _compute_activation_abs_top_stats(flat)
            if tops:
                report["activation_abs"] = tops
            kurt = _global_kurtosis_from_flat(flat)
            if kurt is not None:
                report["kurtosis"] = kurt

        mrr = _compute_max_median_ratio(self._run_max_abs_token_feature_by_layer)
        if mrr is not None:
            report["max_median_ratio"] = mrr

        layer_kurts: list[float] = []
        for state in self._layer_running_stats.values():
            count = int(state.get("count", 0))
            sum_ = state.get("sum")
            sumsq = state.get("sumsq")
            sum3 = state.get("sum3")
            sum4 = state.get("sum4")
            if count <= 0 or not isinstance(sum_, torch.Tensor):
                continue
            layer_stats = _channel_stats_from_running_moments(
                count=count,
                sum_=sum_,
                sumsq=sumsq if isinstance(sumsq, torch.Tensor) else sum_,
                sum3=sum3 if isinstance(sum3, torch.Tensor) else None,
                sum4=sum4 if isinstance(sum4, torch.Tensor) else None,
            )
            for v in layer_stats.get("kurtosis", []):
                if isinstance(v, (int, float)) and v == v:
                    layer_kurts.append(float(v))
        if layer_kurts:
            report["kurtosis_per_channel_mean"] = float(sum(layer_kurts) / len(layer_kurts))
            report["kurtosis_per_channel_max"] = float(max(layer_kurts))

        return report

    def _finalize_on_exit(self) -> None:
        if not self._final_dump_done:
            self._log("finalizing analyzer output")
            self._dump_stats(draw_charts=self.draw_charts)
        self._print_outlier_table(force=True)

    def _merge_modality_attribution(
        self,
        agg: dict[str, Any],
        run: dict[str, Any],
        layer_key: str,
        dim_key: str,
        agg_f: list[bool],
        run_f: list[bool],
        merged_f: list[bool],
        n: int,
    ) -> None:
        def _pad(values: list[Any], fill: Any, size: int) -> list[Any]:
            out = list(values)
            if len(out) < size:
                out.extend([fill] * (size - len(out)))
            return out

        agg_layer = _pad(list(agg.get(layer_key, [])), None, n)
        run_layer = _pad(list(run.get(layer_key, [])), None, n)
        agg_dim = _pad(list(agg.get(dim_key, [])), None, n)
        run_dim = _pad(list(run.get(dim_key, [])), None, n)
        ml: list[Any] = []
        md: list[Any] = []
        for i in range(n):
            if not merged_f[i]:
                ml.append(None)
                md.append(None)
            elif run_f[i]:
                rl, rd = run_layer[i], run_dim[i]
                al, ad = agg_layer[i], agg_dim[i]
                ml.append(rl if rl not in (None, "") else al)
                md.append(rd if rd is not None else ad)
            else:
                ml.append(agg_layer[i])
                md.append(agg_dim[i])
        agg[layer_key] = ml
        agg[dim_key] = md

    def _merge_outliers_across_calls(self, run_outliers: dict[str, Any] | None) -> None:
        if run_outliers is None:
            return
        if self._aggregate_generate_outliers is None:
            self._aggregate_generate_outliers = copy.deepcopy(run_outliers)
            _sync_outlier_attribution_columns(self._aggregate_generate_outliers)
            self._aggregate_generate_outliers["table"] = (
                _format_outlier_table_from_payload(self._aggregate_generate_outliers)
            )
            self._log("initialized outlier aggregation state")
            return

        agg = self._aggregate_generate_outliers
        run_soft_flags = [
            bool(v)
            for v in run_outliers.get("llm.int8() outliers soft difinition", [])
        ]
        agg_soft_flags = [
            bool(v) for v in agg.get("llm.int8() outliers soft difinition", [])
        ]
        run_hard_src = run_outliers.get(
            "llm.int8() outliers hard definition", run_outliers.get("outliers", [])
        )
        agg_hard_src = agg.get(
            "llm.int8() outliers hard definition", agg.get("outliers", [])
        )
        run_hard_flags = [bool(v) for v in run_hard_src]
        agg_hard_flags = [bool(v) for v in agg_hard_src]
        run_iqr_flags = [bool(v) for v in run_outliers.get("interquantile_outliers", [])]
        agg_iqr_flags = [bool(v) for v in agg.get("interquantile_outliers", [])]
        n = max(
            len(agg_hard_flags),
            len(run_hard_flags),
            len(agg_soft_flags),
            len(run_soft_flags),
            len(agg_iqr_flags),
            len(run_iqr_flags),
        )

        def _pad(values: list[Any], fill: Any, size: int) -> list[Any]:
            out = list(values)
            if len(out) < size:
                out.extend([fill] * (size - len(out)))
            return out

        agg_soft_flags = _pad(agg_soft_flags, False, n)
        run_soft_flags = _pad(run_soft_flags, False, n)
        agg_hard_flags = _pad(agg_hard_flags, False, n)
        run_hard_flags = _pad(run_hard_flags, False, n)
        agg_iqr_flags = _pad(agg_iqr_flags, False, n)
        run_iqr_flags = _pad(run_iqr_flags, False, n)

        merged_soft_flags = [
            a or b for a, b in zip(agg_soft_flags, run_soft_flags, strict=False)
        ]
        merged_hard_flags = [
            a or b for a, b in zip(agg_hard_flags, run_hard_flags, strict=False)
        ]
        merged_iqr_flags = [
            a or b for a, b in zip(agg_iqr_flags, run_iqr_flags, strict=False)
        ]
        agg["llm.int8() outliers soft difinition"] = merged_soft_flags
        agg["llm.int8() outliers hard definition"] = merged_hard_flags
        agg["interquantile_outliers"] = merged_iqr_flags
        agg["outliers"] = merged_hard_flags
        agg["token_count"] = int(
            max(int(agg.get("token_count", 0)), int(run_outliers.get("token_count", 0)))
        )
        self._log(f"aggregated outliers across calls (tokens={n})")

        for key, fill in (
            ("prompt_tokens", ""),
            ("prompt_token_ids", ""),
        ):
            agg_vals = _pad(list(agg.get(key, [])), fill, n)
            run_vals = _pad(list(run_outliers.get(key, [])), fill, n)
            merged_vals: list[Any] = []
            for i in range(n):
                if agg_vals[i] not in (None, ""):
                    merged_vals.append(agg_vals[i])
                else:
                    merged_vals.append(run_vals[i])
            agg[key] = merged_vals

        self._merge_modality_attribution(
            agg,
            run_outliers,
            _attribution_layer_key(_SOFT_FLAG),
            _attribution_dim_key(_SOFT_FLAG),
            agg_soft_flags,
            run_soft_flags,
            merged_soft_flags,
            n,
        )
        self._merge_modality_attribution(
            agg,
            run_outliers,
            _attribution_layer_key(_HARD_FLAG),
            _attribution_dim_key(_HARD_FLAG),
            agg_hard_flags,
            run_hard_flags,
            merged_hard_flags,
            n,
        )
        self._merge_modality_attribution(
            agg,
            run_outliers,
            _attribution_layer_key(_IQR_FLAG),
            _attribution_dim_key(_IQR_FLAG),
            agg_iqr_flags,
            run_iqr_flags,
            merged_iqr_flags,
            n,
        )

        agg[_OUTLIERS_DETECTED] = _compute_outliers_detected_list(
            merged_soft_flags,
            merged_hard_flags,
            merged_iqr_flags,
        )

        _sync_outlier_attribution_columns(agg)

        agg["table"] = _format_outlier_table_from_payload(agg)

    def _dump_stats(self, draw_charts: bool = False) -> None:
        stats = self._build_stats()
        report = self._build_activation_report()
        if report:
            stats["report"] = report
        if self._aggregate_generate_outliers is not None:
            self._aggregate_generate_outliers["chart_3d_max_tokens"] = (
                self.chart_3d_max_tokens
            )
            stats["outliers"] = self._aggregate_generate_outliers
        stats["_acta"] = {
            "dump_stats_root": self.dump_stats_root.as_posix(),
            "output_run_dir": self.output_run_dir.as_posix(),
            "dump_stats_path": self.dump_stats_path,
            "session_timestamp": self._session_timestamp,
            "results_csv": (self.dump_stats_root / "acta_results.csv").as_posix(),
            "model_name": _infer_acta_model_name(self.model),
            "draw_token_trends": self.draw_token_trends,
        }
        with open(self.dump_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        self._write_acta_results_csv()
        if draw_charts:
            chart_dir = Path(self.dump_stats_path).parent
            self._log(f"creating charts in: {chart_dir.as_posix()}")
            draw_activation_charts(
                stats=stats,
                output_dir=chart_dir.as_posix(),
                draw_token_trends=self.draw_token_trends,
            )
        self._final_dump_done = bool(draw_charts)
        del stats
        _maybe_release_accelerator_memory()

    def _run_and_dump(self, runner: Any, *args: Any, **kwargs: Any) -> Any:
        result = runner(*args, **kwargs)
        self._dump_stats(draw_charts=False)
        return result

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._reset_run_sequence_buffers()
        self._in_generate = True
        try:
            result = self.model.forward(*args, **kwargs)
        finally:
            self._in_generate = False

        if self._prompt_len is not None:
            use_len = int(self._prompt_len)
            input_ids = kwargs.get("input_ids")
            if (
                isinstance(input_ids, torch.Tensor)
                and input_ids.ndim == 2
                and input_ids.shape[1] >= use_len
            ):
                token_ids = (
                    input_ids[0, :use_len].detach().cpu().to(torch.long).tolist()
                )
            else:
                token_ids = list(range(use_len))
            decoded = (
                self._sequence_position_labels(use_len)
                if self.tokenizer is None
                else [decode_token(self.tokenizer, int(i)) for i in token_ids]
            )
            bsz = self._current_run_batch_size()
            fake = torch.arange(use_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)
            self._last_generate_outliers = self._detect_outliers_after_generate(
                fake, prompt_len=use_len
            )
            if self._last_generate_outliers is not None:
                self._last_generate_outliers["prompt_token_ids"] = token_ids
                self._last_generate_outliers["prompt_tokens"] = decoded
                if "token_trends" in self._last_generate_outliers and isinstance(
                    self._last_generate_outliers["token_trends"], dict
                ):
                    self._last_generate_outliers["token_trends"]["tokens"] = decoded
                    self._last_generate_outliers["token_trends"]["token_ids"] = (
                        token_ids
                    )
                self._last_generate_outliers["table"] = (
                    _format_outlier_table_from_payload(self._last_generate_outliers)
                )

        self._merge_outliers_across_calls(self._last_generate_outliers)
        self._last_generate_outliers = None
        self._dump_stats(draw_charts=False)
        return result

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.model, "generate"):
            raise AttributeError("Wrapped model has no 'generate' method")
        self._reset_run_sequence_buffers()
        self._in_generate = True

        prompt_len: int | None = None
        input_ids = kwargs.get("input_ids", None)
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
            prompt_len = int(input_ids.shape[1])
            self._prompt_len = prompt_len
        try:
            result = self.model.generate(*args, **kwargs)
        finally:
            self._in_generate = False

        if (
            isinstance(result, torch.Tensor)
            and result.ndim == 2
            and prompt_len is not None
        ):
            self._last_prompt_token_ids = result[0, :prompt_len].detach().cpu().tolist()

        self._last_generate_outliers = self._detect_outliers_after_generate(
            result, prompt_len=prompt_len
        )
        self._merge_outliers_across_calls(self._last_generate_outliers)
        self._last_generate_outliers = None
        self._dump_stats(draw_charts=False)
        return result

    def _detect_outliers_after_generate(
        self, generate_result: Any, prompt_len: int | None
    ) -> dict[str, Any] | None:
        """
        Token-level ``outliers`` and attribution columns are **Acta** summaries, not a
        verbatim copy of bitsandbytes LLM.int8() mixed matmul routing.

        Reference (bitsandbytes): `Linear8bitLt` uses `state.threshold` on layer
        inputs; ``bitsandbytes::int8_vectorwise_quant`` builds ``outlier_cols`` where
        `(A.abs() >= threshold).any(dim=0)` over all leading dimensions (see
        `bitsandbytes/backends/default/ops.py`). That is **column-global** per Linear.

        Acta: per hooked layer we compute each token's fraction of hidden dims with
        `|x| >= outlier_threshold`, require `hard_seqdim_frac` per layer, then for
        each token require at least `hard_layers_frac` of layers to pass --- optional
        batch aggregate via `hard_mask.any(dim=0)`. Feature-dimension voting for charts
        uses another aggregation over layers.

        Legacy ``per_token_layer`` / ``per_token_channel_dim`` mirror **hard**
        LLM-style flags only (synced from ``llm.int8() outliers hard definition_*``
        columns). Each detection modality also gets parallel ``*_layer`` /
        ``*_channel_dim`` lists (soft / hard / interquantile). The boolean
        ``outliers detected`` row is true when any modality fires for that token.
        """
        if not isinstance(generate_result, torch.Tensor) or generate_result.ndim != 2:
            return None

        per_layer_frac: dict[str, torch.Tensor] = {}
        for layer_name, chunks in self._run_token_frac_by_layer.items():
            if not chunks:
                continue
            per_layer_frac[layer_name] = torch.cat(chunks, dim=1)  # [B, T_run]

        layer_names = sorted(per_layer_frac.keys())
        if not layer_names:
            return {
                "threshold": self.outlier_threshold,
                "outliers": [],
                "note": "No eligible [B,S,H] activations were captured during generate().",
            }

        b_min = min(t.shape[0] for t in per_layer_frac.values())
        t_min = min(t.shape[1] for t in per_layer_frac.values())
        frac_stack = torch.stack(
            [per_layer_frac[name][:b_min, :t_min] for name in layer_names], dim=0
        )  # [L, B, T]

        layer_affected = frac_stack >= self.hard_seqdim_frac  # [L, B, T]
        frac_layers_affected = layer_affected.to(torch.float32).mean(dim=0)  # [B, T]
        hard_mask = frac_layers_affected >= self.hard_layers_frac
        soft_mask = frac_layers_affected > 0.0

        batch_size = int(generate_result.shape[0])
        hard_any = hard_mask.any(dim=0)  # [T]
        soft_any = soft_mask.any(dim=0)  # [T]

        use_len = min(prompt_len, t_min) if prompt_len is not None else t_min
        soft_outliers_list = soft_any[:use_len].detach().cpu().tolist()
        hard_outliers_list = hard_any[:use_len].detach().cpu().tolist()
        if batch_size == 1:
            token_ids = generate_result[0, :t_min].detach().cpu().tolist()[:use_len]
            decoded_tokens = [decode_token(self.tokenizer, tid) for tid in token_ids]
        else:
            token_ids = list(range(use_len))
            decoded_tokens = self._sequence_position_labels(use_len)

        per_layer_mean: dict[str, torch.Tensor] = {}
        per_layer_var: dict[str, torch.Tensor] = {}
        for layer_name in layer_names:
            mean_chunks = self._run_token_mean_by_layer.get(layer_name, [])
            var_chunks = self._run_token_var_by_layer.get(layer_name, [])
            if not mean_chunks or not var_chunks:
                continue
            per_layer_mean[layer_name] = torch.cat(mean_chunks, dim=1)[
                :b_min, :t_min
            ]  # [B, T]
            per_layer_var[layer_name] = torch.cat(var_chunks, dim=1)[
                :b_min, :t_min
            ]  # [B, T]

        token_trends: dict[str, Any] | None = None
        if len(per_layer_mean) == len(layer_names):
            means_layer_token = []
            vars_layer_token = []
            for ln in layer_names:
                means_layer_token.append(
                    per_layer_mean[ln][:, :use_len].mean(dim=0).cpu().tolist()
                )
                vars_layer_token.append(
                    per_layer_var[ln][:, :use_len].mean(dim=0).cpu().tolist()
                )
            token_trends = {
                "layer_names": layer_names,
                "token_count": int(use_len),
                # shape: [num_layers][token_count]
                "mean": means_layer_token,
                "variance": vars_layer_token,
                "tokens": decoded_tokens,
                "token_ids": token_ids,
            }

        outlier_feature_dims: list[int] = []
        h_shared: int | None = None
        if self._run_feature_counts_by_layer and self._run_feature_token_count_by_layer:
            eligible_layers = [
                ln for ln in layer_names if ln in self._run_feature_counts_by_layer
            ]
            if eligible_layers:
                lengths = [
                    int(self._run_feature_counts_by_layer[ln].shape[0])
                    for ln in eligible_layers
                ]
                h = int(max(set(lengths), key=lengths.count))
                eligible_layers = [
                    ln
                    for ln in eligible_layers
                    if int(self._run_feature_counts_by_layer[ln].shape[0]) == h
                ]
                h_shared = h
                affected_layers_per_dim = torch.zeros(h, dtype=torch.float32)
                num_layers = len(eligible_layers)
                if num_layers > 0:
                    for ln in eligible_layers:
                        total_toks = max(
                            int(self._run_feature_token_count_by_layer.get(ln, 0)), 1
                        )
                        frac_tokens = self._run_feature_counts_by_layer[ln].to(
                            torch.float32
                        ) / float(total_toks)
                        affected_layers_per_dim += (
                            frac_tokens >= float(self.hard_seqdim_frac)
                        ).to(torch.float32)
                    frac_layers = affected_layers_per_dim / float(num_layers)
                    outlier_feature_dims = (
                        (frac_layers >= float(self.hard_layers_frac))
                        .nonzero(as_tuple=False)
                        .view(-1)
                        .cpu()
                        .tolist()
                    )

        token_feature_magnitude: list[list[float]] | None = None
        if self._run_max_abs_token_feature is not None and outlier_feature_dims:
            t_feat = min(int(use_len), int(self._run_max_abs_token_feature.shape[0]))
            max_abs = self._run_max_abs_token_feature[:t_feat, :]  # [T, H]
            token_feature_magnitude = (
                max_abs[:, outlier_feature_dims].cpu().tolist()
            )  # [T][F]

        token_feature_magnitude_by_layer: dict[str, list[list[float]]] = {}
        h_for_slices = h_shared
        if h_for_slices is None and self._run_max_abs_token_feature is not None:
            h_for_slices = int(self._run_max_abs_token_feature.shape[1])
        if (
            outlier_feature_dims
            and self._run_max_abs_token_feature_by_layer
            and h_for_slices is not None
        ):
            for ln, ten in self._run_max_abs_token_feature_by_layer.items():
                if ten.ndim != 2:
                    continue
                if int(ten.shape[1]) != int(h_for_slices):
                    continue
                t_use = min(int(use_len), int(ten.shape[0]))
                if t_use <= 0:
                    continue
                token_feature_magnitude_by_layer[ln] = (
                    ten[:t_use, outlier_feature_dims].cpu().tolist()
                )

        soft_layer_attr: list[str | None] = [None] * int(use_len)
        soft_dim_attr: list[int | None] = [None] * int(use_len)
        neg_one_fs = torch.tensor(
            -1.0, device=frac_stack.device, dtype=frac_stack.dtype
        )
        for ti in range(int(use_len)):
            if not soft_outliers_list[ti]:
                continue
            sm_batch = soft_mask[:, ti]
            if not bool(sm_batch.any().item()):
                continue
            masked_scores = torch.where(
                sm_batch, frac_layers_affected[:, ti], neg_one_fs
            )
            b_pick = int(torch.argmax(masked_scores).item())
            fr = frac_stack[:, b_pick, ti]
            pos = fr > 0
            if not bool(pos.any().item()):
                continue
            masked_fr = torch.where(pos, fr, neg_one_fs)
            l_pick = int(torch.argmax(masked_fr).item())
            ln = layer_names[l_pick]
            soft_layer_attr[ti] = ln
            ten_soft = self._run_max_abs_token_feature_by_layer.get(ln)
            if ten_soft is not None and ti < ten_soft.shape[0]:
                soft_dim_attr[ti] = int(torch.argmax(ten_soft[ti]).item())

        interquantile_outliers_list: list[bool] = [False] * int(use_len)
        iqr_mask_full: torch.Tensor | None = None
        max_abs_for_attrib: torch.Tensor | None = None
        if self._run_max_abs_token_feature is not None:
            t_feat = min(int(use_len), int(self._run_max_abs_token_feature.shape[0]))
            if t_feat > 0:
                max_abs = self._run_max_abs_token_feature[:t_feat, :]  # [T, H]
                max_abs_for_attrib = max_abs
                q1 = torch.quantile(max_abs, 0.25, dim=0)
                q3 = torch.quantile(max_abs, 0.75, dim=0)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                iqr_mask = (max_abs < lower) | (max_abs > upper)  # [T, H]
                iqr_mask_full = iqr_mask
                interquantile_outliers_list = iqr_mask.any(dim=1).cpu().tolist()

        iqr_layer_attr: list[str | None] = [None] * int(use_len)
        iqr_dim_attr: list[int | None] = [None] * int(use_len)
        if max_abs_for_attrib is not None and iqr_mask_full is not None:
            by_layer = self._run_max_abs_token_feature_by_layer
            n_abs = int(max_abs_for_attrib.shape[0])
            neg_ma = torch.tensor(
                -1.0,
                device=max_abs_for_attrib.device,
                dtype=max_abs_for_attrib.dtype,
            )
            for ti in range(min(int(use_len), n_abs)):
                if interquantile_outliers_list[ti]:
                    row_iqr = iqr_mask_full[ti]
                    if bool(row_iqr.any().item()):
                        vals_iqr = torch.where(
                            row_iqr, max_abs_for_attrib[ti], neg_ma
                        )
                        h_iqr = int(torch.argmax(vals_iqr).item())
                        iqr_dim_attr[ti] = h_iqr
                        iqr_layer_attr[ti] = _best_layer_for_hidden_dim(
                            by_layer, ti, h_iqr
                        )

        per_token_layer: list[str | None] = [None] * int(use_len)
        per_token_channel_dim: list[int | None] = [None] * int(use_len)
        if self._run_max_abs_token_feature is not None and outlier_feature_dims:
            t_feat = min(int(use_len), int(self._run_max_abs_token_feature.shape[0]))
            max_abs = self._run_max_abs_token_feature[:t_feat, :]  # [T, H]
            for ti in range(int(t_feat)):
                if ti >= len(hard_outliers_list) or not hard_outliers_list[ti]:
                    continue
                vals = max_abs[ti, outlier_feature_dims]
                if vals.numel() == 0:
                    continue
                j = int(torch.argmax(vals).item())
                per_token_channel_dim[ti] = int(outlier_feature_dims[j])
                if (
                    self._run_argmax_layer_per_token_feature is not None
                    and self._run_layer_order
                ):
                    li = int(
                        self._run_argmax_layer_per_token_feature[
                            ti, per_token_channel_dim[ti]
                        ].item()
                    )
                    if 0 <= li < len(self._run_layer_order):
                        per_token_layer[ti] = self._run_layer_order[li]

        token_layer_max_magnitude: list[list[float]] | None = None
        if self._run_token_maxabs_by_layer:
            layer_list = sorted(self._run_token_maxabs_by_layer.keys())
            t_layer_min = min(
                torch.cat(chunks, dim=0).shape[0]
                for chunks in self._run_token_maxabs_by_layer.values()
            )
            t_layer_use = min(int(use_len), int(t_layer_min))
            token_layer_max_magnitude = []
            for ln in layer_list:
                seq = torch.cat(self._run_token_maxabs_by_layer[ln], dim=0)[
                    :t_layer_use
                ]  # [T]
                token_layer_max_magnitude.append(seq.cpu().tolist())

        outliers_detected_list = _compute_outliers_detected_list(
            soft_outliers_list,
            hard_outliers_list,
            interquantile_outliers_list,
        )

        payload_outliers: dict[str, Any] = {
            _SOFT_FLAG: soft_outliers_list,
            _HARD_FLAG: hard_outliers_list,
            _IQR_FLAG: interquantile_outliers_list,
            "outliers": hard_outliers_list,
            "per_token_layer": per_token_layer,
            "per_token_channel_dim": per_token_channel_dim,
            _attribution_layer_key(_SOFT_FLAG): soft_layer_attr,
            _attribution_dim_key(_SOFT_FLAG): soft_dim_attr,
            _attribution_layer_key(_HARD_FLAG): per_token_layer,
            _attribution_dim_key(_HARD_FLAG): per_token_channel_dim,
            _attribution_layer_key(_IQR_FLAG): iqr_layer_attr,
            _attribution_dim_key(_IQR_FLAG): iqr_dim_attr,
            _OUTLIERS_DETECTED: outliers_detected_list,
        }
        _sync_outlier_attribution_columns(payload_outliers)

        tbl_src = {
            **payload_outliers,
            "prompt_tokens": decoded_tokens,
            "prompt_token_ids": token_ids,
        }

        return {
            "threshold": self.outlier_threshold,
            "hard_criteria": {
                "layers_fraction_threshold": self.hard_layers_frac,
                "seqdim_fraction_threshold": self.hard_seqdim_frac,
            },
            "num_layers_considered": len(layer_names),
            "token_count": int(use_len),
            _SOFT_FLAG: soft_outliers_list,
            _HARD_FLAG: hard_outliers_list,
            _IQR_FLAG: interquantile_outliers_list,
            "outliers": hard_outliers_list,
            "prompt_token_ids": token_ids,
            "prompt_tokens": decoded_tokens,
            "per_token_layer": payload_outliers["per_token_layer"],
            "per_token_channel_dim": payload_outliers["per_token_channel_dim"],
            _attribution_layer_key(_SOFT_FLAG): soft_layer_attr,
            _attribution_dim_key(_SOFT_FLAG): soft_dim_attr,
            _attribution_layer_key(_HARD_FLAG): payload_outliers[
                _attribution_layer_key(_HARD_FLAG)
            ],
            _attribution_dim_key(_HARD_FLAG): payload_outliers[
                _attribution_dim_key(_HARD_FLAG)
            ],
            _attribution_layer_key(_IQR_FLAG): iqr_layer_attr,
            _attribution_dim_key(_IQR_FLAG): iqr_dim_attr,
            _OUTLIERS_DETECTED: outliers_detected_list,
            "table": _format_outlier_table_from_payload(tbl_src),
            "token_trends": token_trends,
            "outlier_feature_dims": outlier_feature_dims,
            "token_feature_magnitude": token_feature_magnitude,
            "token_feature_magnitude_by_layer": token_feature_magnitude_by_layer,
            "token_layer_trends": {
                "layer_names": sorted(self._run_token_maxabs_by_layer.keys()),
                "token_count": int(use_len),
                "tokens": decoded_tokens,
                "token_ids": token_ids,
                # shape: [num_layers][token_count]
                "max_abs": token_layer_max_magnitude,
            },
        }

    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.model, item)


def AutoAnalyzer(
    model: nn.Module,
    dump_stats_path: str | None = None,
    target_layers: str | list[str] | None = None,
    draw_charts: bool = False,
    draw_token_trends: bool = False,
    outlier_threshold: float = 6.0,
    hard_layers_frac: float = 0.25,
    hard_seqdim_frac: float = 0.06,
    verbose: bool = False,
    tokenizer: Any | None = None,
    vit_reg_patch_labels: bool = False,
    asr_chunk_labels: bool = False,
    chart_3d_max_tokens: int = 24,
    finalize_on_exit: bool = True,
) -> nn.Module:
    return _AnalyzerModel(
        model=model,
        dump_stats_path=dump_stats_path,
        target_layers=target_layers,
        draw_charts=draw_charts,
        draw_token_trends=draw_token_trends,
        outlier_threshold=outlier_threshold,
        hard_layers_frac=hard_layers_frac,
        hard_seqdim_frac=hard_seqdim_frac,
        verbose=verbose,
        tokenizer=tokenizer,
        vit_reg_patch_labels=vit_reg_patch_labels,
        asr_chunk_labels=asr_chunk_labels,
        chart_3d_max_tokens=chart_3d_max_tokens,
        finalize_on_exit=finalize_on_exit,
    )
