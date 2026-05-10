from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch
from dash import Dash, Input, Output, State, dash_table, dcc, html
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from acta import AutoAnalyzer
_LOG_PATH = (Path.cwd() / ".acta_webui.log").resolve()

_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "source": None,
    "task": None,
    "device": torch.device("cpu"),
}

logger = logging.getLogger("acta.webui")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.addHandler(_fh)
    logger.propagate = False


def _log(msg: str, *args: Any) -> None:
    logger.info(msg, *args)


_GRAPH_STYLE: dict[str, str | int] = {
    "height": "520px",
    "minHeight": "480px",
    "width": "100%",
}


def _hover_plain(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _tensor_brief(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
    return type(v).__name__


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_nn_module(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        for key in ("model", "module", "student", "teacher", "net"):
            cand = obj.get(key)
            if isinstance(cand, nn.Module):
                return cand
        for cand in obj.values():
            if isinstance(cand, nn.Module):
                return cand
    raise ValueError("Could not find nn.Module in checkpoint.")


def _convert_leaf(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _convert_leaf(x) for k, x in v.items()}
    if isinstance(v, list):
        try:
            return torch.tensor(v)
        except Exception:
            return [_convert_leaf(x) for x in v]
    return v


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=False)
        else:
            out[k] = v
    return out


def _sample_payload(source: str, task: str) -> str:
    if source == "hf" and task == "causal-lm":
        return "Hello, this is a sample prompt"
    if source == "hf":
        return "This movie is fantastic and emotionally engaging."
    return json.dumps(
        {"input": [[0.1, -0.2, 0.3, 0.4]]},
        indent=2,
    )


def _csv_for_dash(csv_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not csv_path.exists():
        return [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = [{"name": name, "id": name} for name in (reader.fieldnames or [])]
    return rows, cols


def _fig_outlier_token_feature_3d(stats: dict[str, Any]) -> go.Figure:
    out = stats.get("outliers", {})
    if not isinstance(out, dict):
        return _empty_fig("No outlier payload")
    z = out.get("token_feature_magnitude")
    feat_dims = out.get("outlier_feature_dims", [])
    tokens = out.get("prompt_tokens", [])
    if not isinstance(z, list) or not z or not feat_dims:
        return _empty_fig("No outlier token feature 3D data")
    z_np = np.asarray(z, dtype=np.float64)
    if z_np.ndim != 2:
        return _empty_fig("Malformed token_feature_magnitude shape")
    n_tok, n_feat = int(z_np.shape[0]), int(z_np.shape[1])
    fd_raw = feat_dims if isinstance(feat_dims, list) else []
    fd_use: list[int] = []
    for j in range(n_feat):
        if j < len(fd_raw):
            fd_use.append(int(fd_raw[j]))
        else:
            fd_use.append(j)
    tok_list = tokens if isinstance(tokens, list) else []
    xi = np.arange(n_feat, dtype=np.float64)
    yi = np.arange(n_tok, dtype=np.float64)
    hover: list[list[str]] = []
    for ti in range(n_tok):
        tok_s = _hover_plain(str(tok_list[ti]) if ti < len(tok_list) else f"<idx {ti}>")
        hover.append(
            [
                f"token idx {ti}: {tok_s}<br>feature dim {fd_use[fj]}<br>magnitude: {float(z_np[ti, fj]):.6g}"
                for fj in range(n_feat)
            ]
        )
    fig = go.Figure(
        data=[
            go.Surface(
                x=xi,
                y=yi,
                z=z_np,
                surfacecolor=z_np,
                colorscale="Viridis",
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="outlier token feature 3d",
        scene=dict(
            xaxis_title="Feature dim index",
            yaxis_title="Token index",
            zaxis_title="Magnitude",
            xaxis=dict(tickmode="linear"),
            yaxis=dict(tickmode="linear"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return _apply_dark(fig)


def _fig_token_layer_maxabs_3d(stats: dict[str, Any]) -> go.Figure:
    out = stats.get("outliers", {})
    trends = out.get("token_layer_trends", {}) if isinstance(out, dict) else {}
    if not isinstance(trends, dict):
        return _empty_fig("No token layer trend data")
    z = trends.get("max_abs")
    layers = trends.get("layer_names", [])
    tokens = trends.get("tokens", [])
    if not isinstance(z, list) or not z:
        return _empty_fig("No token layer maxabs 3D data")
    z_np = np.asarray(z, dtype=np.float64)
    if z_np.ndim != 2:
        return _empty_fig("Malformed token layer maxabs shape")
    n_layer, n_tok = int(z_np.shape[0]), int(z_np.shape[1])
    ly = layers if isinstance(layers, list) else []
    tok_list = tokens if isinstance(tokens, list) else []
    xi = np.arange(n_tok, dtype=np.float64)
    yi = np.arange(n_layer, dtype=np.float64)
    hover_ll: list[list[str]] = []
    for li in range(n_layer):
        ln = str(ly[li]) if li < len(ly) else f"layer[{li}]"
        hover_ll.append(
            [
                f"{ln}<br>token idx {tj}: {_hover_plain(str(tok_list[tj]) if tj < len(tok_list) else str(tj))}<br>max |act|: {float(z_np[li, tj]):.6g}"
                for tj in range(n_tok)
            ]
        )
    fig = go.Figure(
        data=[
            go.Surface(
                x=xi,
                y=yi,
                z=z_np,
                surfacecolor=z_np,
                colorscale="Plasma",
                hovertext=hover_ll,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="token layer maxabs 3d",
        scene=dict(
            xaxis_title="Token index",
            yaxis_title="Layer index",
            zaxis_title="Max |activation|",
            xaxis=dict(tickmode="linear"),
            yaxis=dict(tickmode="linear"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return _apply_dark(fig)


def _fig_feature_magnitudes_per_layer(stats: dict[str, Any]) -> list[Any]:
    out = stats.get("outliers", {})
    by_layer = out.get("token_feature_magnitude_by_layer", {}) if isinstance(out, dict) else {}
    feat_dims = out.get("outlier_feature_dims", []) if isinstance(out, dict) else []
    if not isinstance(by_layer, dict) or not by_layer:
        return [html.P("No per-layer feature magnitude data", className="acta-sub")]
    items: list[Any] = []
    for layer_name, data in sorted(by_layer.items()):
        if not isinstance(data, list) or not data:
            continue
        z_np = np.asarray(data, dtype=np.float64)
        if z_np.ndim != 2:
            continue
        n_tok, n_feat = int(z_np.shape[0]), int(z_np.shape[1])
        fd_raw = feat_dims if isinstance(feat_dims, list) else []
        fd_use_pl: list[int] = []
        for j in range(n_feat):
            if j < len(fd_raw):
                fd_use_pl.append(int(fd_raw[j]))
            else:
                fd_use_pl.append(j)
        xi = np.arange(n_feat, dtype=np.float64)
        yi = np.arange(n_tok, dtype=np.float64)
        hover_pl: list[list[str]] = []
        for ti in range(n_tok):
            hover_pl.append(
                [
                    f"{layer_name}<br>token idx {ti}<br>feature dim {fd_use_pl[fj]}<br>magnitude: {float(z_np[ti, fj]):.6g}"
                    for fj in range(n_feat)
                ]
            )
        fig = go.Figure(
            data=[
                go.Surface(
                    x=xi,
                    y=yi,
                    z=z_np,
                    surfacecolor=z_np,
                    colorscale="Cividis",
                    hovertext=hover_pl,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"PER_LAYER: {layer_name}",
            scene=dict(
                xaxis_title="Feature dim index",
                yaxis_title="Token index",
                zaxis_title="Magnitude",
                xaxis=dict(tickmode="linear"),
                yaxis=dict(tickmode="linear"),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        items.append(
            html.Div(
                [
                    dcc.Graph(
                        figure=_apply_dark(fig),
                        config={"scrollZoom": True},
                        style=dict(_GRAPH_STYLE),
                    )
                ],
                className="acta-chart-card",
            )
        )
    if not items:
        return [html.P("No per-layer feature magnitude data", className="acta-sub")]
    return items


def _apply_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, system-ui, sans-serif"},
    )
    return fig


def _empty_fig(title: str) -> go.Figure:
    return _apply_dark(go.Figure(layout={"title": title}))


def _fig_layer_means(stats: dict[str, Any]) -> go.Figure:
    layers = stats.get("layers", {})
    names: list[str] = []
    vals: list[float] = []
    if isinstance(layers, dict):
        for ln, st in layers.items():
            m = st.get("mean", [])
            if not isinstance(m, list) or not m:
                continue
            nums: list[float] = []
            for x in m:
                if isinstance(x, bool):
                    continue
                if isinstance(x, (int, float)):
                    v = float(x)
                    if v == v:
                        nums.append(v)
            if not nums:
                continue
            names.append(str(ln))
            vals.append(sum(nums) / len(nums))
    if not names:
        fig = _empty_fig("No per-layer mean data (run with targeted layers or check hooks)")
        fig.update_layout(height=420)
        return fig
    pairs = sorted(zip(names, vals), key=lambda nv: nv[1])
    names_o = [p[0] for p in pairs]
    vals_o = [p[1] for p in pairs]
    nh = len(names_o)
    fig = go.Figure(
        data=[
            go.Bar(
                x=vals_o,
                y=names_o,
                orientation="h",
                marker_color="#38bdf8",
            )
        ]
    )
    fig.update_layout(
        title="Per-layer mean activations",
        xaxis_title="Mean activation (scalar average over channels)",
        yaxis_title="Layer",
        height=max(440, min(960, 26 * nh + 160)),
        margin=dict(l=8, r=16, t=48, b=48),
        autosize=True,
    )
    return _apply_dark(fig)


def _fig_token_layer_heatmap(stats: dict[str, Any]) -> go.Figure:
    out = stats.get("outliers", {})
    tr_raw = out.get("token_trends", {}) if isinstance(out, dict) else {}
    tr = tr_raw if isinstance(tr_raw, dict) else {}
    z = tr.get("mean", [])
    y = tr.get("layer_names", [])
    x = tr.get("tokens", [])
    if not z:
        return _empty_fig("No token trends available")
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar={"title": "mean"},
        )
    )
    fig.update_layout(title="Token trends (mean) heatmap")
    return _apply_dark(fig)


def _fig_outlier_flags(stats: dict[str, Any]) -> go.Figure:
    out_raw = stats.get("outliers", {})
    out = out_raw if isinstance(out_raw, dict) else {}
    if not isinstance(out, dict):
        return _empty_fig("No outlier payload")
    x = list(range(int(out.get("token_count", 0))))
    hard = [
        1 if bool(v) else 0
        for v in out.get("llm.int8() outliers hard definition", [])
    ]
    soft = [
        1 if bool(v) else 0
        for v in out.get("llm.int8() outliers soft difinition", [])
    ]
    iqr = [1 if bool(v) else 0 for v in out.get("interquantile_outliers", [])]
    massive = [1 if bool(v) else 0 for v in out.get("massive_activations", [])]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=soft, mode="lines+markers", name="soft", line={"color": "#94a3b8"})
    )
    fig.add_trace(
        go.Scatter(x=x, y=hard, mode="lines+markers", name="hard", line={"color": "#f87171"})
    )
    fig.add_trace(
        go.Scatter(x=x, y=iqr, mode="lines+markers", name="iqr", line={"color": "#a78bfa"})
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=massive, mode="lines+markers", name="massive", line={"color": "#34d399"}
        )
    )
    fig.update_layout(title="Outlier flags per token", yaxis={"tickvals": [0, 1]})
    return _apply_dark(fig)


def create_app() -> Dash:
    pio.templates.default = "plotly_dark"

    app = Dash(
        __name__,
        assets_folder=str(_assets_dir()),
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Roboto+Serif:ital,opsz,wght@0,8..144,100..900;1,8..144,100..900&display=swap",
        ],
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        [
            dcc.Store(id="results-ready", data=False),
            html.Div(
                [
                    html.H2("ACTA", className="acta-title"),
                    html.P(
                        "Load a model, then run activation analysis with interactive charts.",
                        className="acta-sub",
                    ),
                    html.Div(
                        [
                            html.P("Model source", className="acta-label"),
                            dcc.RadioItems(
                                id="source",
                                options=[
                                    {"label": " HuggingFace ", "value": "hf"},
                                    {"label": " Local .pt ", "value": "local"},
                                ],
                                value="hf",
                                inline=True,
                                className="acta-row",
                            ),
                            html.P("Task", className="acta-label"),
                            dcc.Dropdown(
                                id="task",
                                options=[
                                    {"label": "Causal LM (generate)", "value": "causal-lm"},
                                    {"label": "Encoder (forward)", "value": "encoder"},
                                ],
                                value="causal-lm",
                                clearable=False,
                            ),
                            html.P("HuggingFace model id", className="acta-label"),
                            dcc.Input(
                                id="hf_name",
                                type="text",
                                placeholder="e.g. gpt2",
                                value="gpt2",
                                style={"width": "100%", "marginTop": "4px"},
                            ),
                            html.P("Local checkpoint path", className="acta-label"),
                            dcc.Input(
                                id="local_path",
                                type="text",
                                placeholder="/path/to/model.pt",
                                style={"width": "100%", "marginTop": "4px"},
                            ),
                            html.P("Target layers (optional)", className="acta-label"),
                            dcc.Input(
                                id="target_layers",
                                type="text",
                                placeholder="e.g. transformer.h.*.mlp.c_fc",
                                style={"width": "100%", "marginTop": "4px"},
                            ),
                            html.P("Input payload", className="acta-label"),
                            html.Button(
                                "Suggest sample input",
                                id="btn_suggest",
                                n_clicks=0,
                                className="acta-btn acta-btn-muted",
                                style={"marginTop": "6px"},
                            ),
                            dcc.Textarea(
                                id="payload",
                                value=_sample_payload("hf", "causal-lm"),
                                style={
                                    "width": "100%",
                                    "height": "120px",
                                    "marginTop": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Run analysis",
                                        id="btn_run",
                                        n_clicks=0,
                                        className="acta-btn acta-btn-run",
                                    ),
                                ],
                                className="acta-row",
                                style={"marginTop": "14px"},
                            ),
                            html.Div(
                                [
                                    html.P("Progress", className="acta-label"),
                                    html.P(
                                        "Waiting",
                                        id="progress-title",
                                        className="acta-sub",
                                        style={"margin": "0 0 6px"},
                                    ),
                                    html.Progress(
                                        id="progress",
                                        value=0,
                                        max=100,
                                        style={"width": "100%", "display": "block"},
                                    ),
                                ]
                            ),
                            html.Pre(id="status", className="acta-status"),
                        ],
                        className="acta-card",
                    ),
                ],
                className="acta-col-left",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Tabs(
                                id="results-tabs",
                                value="csv",
                                children=[
                                    dcc.Tab(
                                        label="CSV Results",
                                        value="csv",
                                        children=[
                                            dash_table.DataTable(
                                                id="csv-table",
                                                columns=[],
                                                data=[],
                                                style_table={"overflowX": "auto"},
                                                style_header={
                                                    "backgroundColor": "rgba(56,189,248,0.18)",
                                                    "color": "#e8ecf4",
                                                    "fontWeight": "700",
                                                    "border": "1px solid rgba(148,163,184,0.22)",
                                                },
                                                style_cell={
                                                    "backgroundColor": "rgba(15,23,42,0.5)",
                                                    "color": "#e8ecf4",
                                                    "border": "1px solid rgba(148,163,184,0.16)",
                                                    "fontFamily": "JetBrains Mono, monospace",
                                                    "fontSize": "12px",
                                                    "padding": "6px",
                                                    "textAlign": "left",
                                                    "maxWidth": 260,
                                                    "whiteSpace": "normal",
                                                },
                                                page_size=15,
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="outlier token feature 3d",
                                        value="outlier3d",
                                        children=[
                                            dcc.Graph(
                                                id="fig_outlier3d",
                                                config={"scrollZoom": True},
                                                style=dict(_GRAPH_STYLE),
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="token layer maxabs 3d",
                                        value="layermax3d",
                                        children=[
                                            dcc.Graph(
                                                id="fig_layermax3d",
                                                config={"scrollZoom": True},
                                                style=dict(_GRAPH_STYLE),
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="feature magnitudes per-layer",
                                        value="perlayer3d",
                                        children=[html.Div(id="tab-perlayer3d", className="acta-chart-grid")],
                                    ),
                                    dcc.Tab(
                                        label="per-layer mean activations",
                                        value="means",
                                        children=[
                                            dcc.Graph(
                                                id="fig_layer",
                                                config={"scrollZoom": True},
                                                style=dict(_GRAPH_STYLE),
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="token trends (mean) heatmap",
                                        value="heatmap",
                                        children=[
                                            dcc.Graph(
                                                id="fig_heatmap",
                                                config={"scrollZoom": True},
                                                style=dict(_GRAPH_STYLE),
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        className="acta-card",
                    )
                ],
                className="acta-col-right",
                id="right-pane",
                style={"display": "none"},
            ),
        ]
        ,
        className="acta-shell acta-layout",
    )

    @app.callback(
        Output("right-pane", "style"),
        Input("results-ready", "data"),
    )
    def toggle_right_pane(ready: bool) -> dict[str, Any]:
        if bool(ready):
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("payload", "value"),
        Input("btn_suggest", "n_clicks"),
        State("source", "value"),
        State("task", "value"),
        prevent_initial_call=True,
    )
    def suggest_payload(_: int, source: str, task: str) -> str:
        return _sample_payload(source, task)

    def _model_key(source: str, task: str, hf_name: str, local_path: str) -> str:
        if source == "hf":
            return f"hf::{task}::{(hf_name or '').strip()}"
        return f"local::{task}::{Path((local_path or '').strip()).expanduser().resolve()}"

    def _ensure_model_loaded(
        source: str, task: str, hf_name: str, local_path: str
    ) -> tuple[str, int, str]:
        _log("load_model start source=%s task=%s pid=%s", source, task, os.getpid())
        try:
            device = _preferred_device()
            _log("selected device=%s", device)
            key = _model_key(source, task, hf_name, local_path)
            if _STATE.get("model") is not None and _STATE.get("model_key") == key:
                _log("reuse loaded model key=%s", key)
                return "Stage 1/3 complete: model reused", 33, "Model reused from memory."
            if source == "hf":
                name = (hf_name or "").strip()
                if not name:
                    _log("load_model missing hf name")
                    return "Load failed", 0, "Please provide a HuggingFace model name."
                _log("loading hf model name=%s task=%s", name, task)
                if task == "causal-lm":
                    model = AutoModelForCausalLM.from_pretrained(
                        name, dtype=torch.float32
                    )
                else:
                    model = AutoModel.from_pretrained(name, dtype=torch.float32)
                model = model.to(device).eval()
                _log("model loaded class=%s device=%s", model.__class__.__name__, device)
                tokenizer = AutoTokenizer.from_pretrained(name)
                _log("tokenizer loaded class=%s", tokenizer.__class__.__name__)
                _STATE.update(
                    {
                        "model": model,
                        "tokenizer": tokenizer,
                        "source": source,
                        "task": task,
                        "device": device,
                        "model_key": key,
                    }
                )
                _log("load_model success hf")
                return (
                    "Stage 1/3 complete: model loaded",
                    33,
                    f"Loaded HuggingFace model on {device}: {name} ({task}).",
                )
            path = Path((local_path or "").strip()).expanduser().resolve()
            if not path.exists():
                _log("load_model local missing path=%s", path)
                return "Load failed", 0, f"Local file not found: {path}"
            _log("loading local checkpoint path=%s", path)
            payload = _torch_load(path)
            model = _extract_nn_module(payload).to(device).eval()
            _log("local model loaded class=%s", model.__class__.__name__)
            _STATE.update(
                {
                    "model": model,
                    "tokenizer": None,
                    "source": source,
                    "task": task,
                    "device": device,
                    "model_key": key,
                }
            )
            _log("load_model success local")
            return (
                "Stage 1/3 complete: model loaded",
                33,
                f"Loaded local model on {device} from:\n{path}",
            )
        except Exception as e:
            _log("load_model failed error=%s\n%s", e, traceback.format_exc())
            return "Load failed", 0, f"Load failed: {e}"

    @app.callback(
        Output("fig_layer", "figure"),
        Output("fig_heatmap", "figure"),
        Output("fig_outlier3d", "figure"),
        Output("fig_layermax3d", "figure"),
        Output("csv-table", "data"),
        Output("csv-table", "columns"),
        Output("tab-perlayer3d", "children"),
        Output("results-ready", "data"),
        Output("status", "children", allow_duplicate=True),
        Output("progress-title", "children", allow_duplicate=True),
        Output("progress", "value", allow_duplicate=True),
        Input("btn_run", "n_clicks"),
        State("payload", "value"),
        State("target_layers", "value"),
        State("source", "value"),
        State("task", "value"),
        State("hf_name", "value"),
        State("local_path", "value"),
        running=[
            (
                Output("progress-title", "children"),
                "Stage 2/3: running inference...",
                "Stage 1/3 complete: model loaded",
            ),
            (Output("progress", "value"), 66, 33),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(
        _: int,
        payload_text: str,
        target_layers: str | None,
        source: str,
        task: str,
        hf_name: str,
        local_path: str,
    ) -> tuple[
        Any,
        Any,
        Any,
        Any,
        list[dict[str, Any]],
        list[dict[str, str]],
        list[Any],
        bool,
        str,
        str,
        int,
    ]:
        load_title, load_val, load_msg = _ensure_model_loaded(
            source, task, hf_name, local_path
        )
        model = _STATE.get("model")
        if model is None:
            empty = _empty_fig("No model loaded")
            return (
                empty,
                empty,
                empty,
                empty,
                [],
                [],
                [],
                False,
                load_msg,
                "Stage 1/3 failed: model load",
                0,
            )
        tokenizer = _STATE.get("tokenizer")
        device = _STATE.get("device", torch.device("cpu"))
        _log(
            "run_analysis start source=%s task=%s model=%s device=%s payload_chars=%d",
            source,
            task,
            model.__class__.__name__,
            device,
            len(payload_text or ""),
        )
        wrapped = None
        try:
            tmp = Path(tempfile.mkdtemp(prefix="acta-ui-"))
            _log("temp run dir=%s", tmp)
            wrapped = AutoAnalyzer(
                model,
                dump_stats_path=str(tmp),
                draw_charts=False,
                verbose=False,
                tokenizer=tokenizer,
                target_layers=(target_layers.strip() if target_layers and target_layers.strip() else None),
                finalize_on_exit=False,
            )
            wrapped = wrapped.to(device)
            wrapped.eval()
            _log("wrapped model ready class=%s", wrapped.__class__.__name__)
            with torch.inference_mode():
                if source == "hf":
                    text = (payload_text or "").strip()
                    if not text:
                        _log("run_analysis hf empty payload")
                        raise ValueError("Payload text is empty.")
                    assert tokenizer is not None
                    _log("tokenizing text len=%d", len(text))
                    inputs = tokenizer(text, return_tensors="pt")
                    inputs = _to_device_batch(dict(inputs), device=device)
                    _log(
                        "tokenized keys=%s summary=%s",
                        list(inputs.keys()),
                        {k: _tensor_brief(v) for k, v in inputs.items()},
                    )
                    _log("calling forward on wrapped model (safe mode, no generate)")
                    wrapped(**inputs)
                    _log("forward returned successfully")
                else:
                    _log("parsing local payload json")
                    obj = json.loads(payload_text or "{}")
                    converted = _convert_leaf(obj)
                    if isinstance(converted, dict):
                        merged: dict[str, Any] = {}
                        for k, v in converted.items():
                            if isinstance(v, torch.Tensor):
                                merged[k] = v.to(device)
                            else:
                                merged[k] = v
                        _log(
                            "local input keys=%s summary=%s",
                            list(merged.keys()),
                            {k: _tensor_brief(v) for k, v in merged.items()},
                        )
                        wrapped(**merged)
                        _log("local forward returned successfully")
                    elif isinstance(converted, torch.Tensor):
                        _log("local tensor input summary=%s", _tensor_brief(converted))
                        wrapped(converted.to(device))
                        _log("local tensor forward returned successfully")
                    else:
                        raise ValueError(
                            "Local payload must be JSON object or tensor-like list."
                        )
            _log("reading stats from %s", wrapped.dump_stats_path)
            stats = json.loads(Path(wrapped.dump_stats_path).read_text(encoding="utf-8"))
            run_dir = Path(stats.get("_acta", {}).get("output_run_dir", tmp))
            _log("stage 3/3: building interactive charts")
            csv_rows, csv_cols = _csv_for_dash(run_dir / "acta_results.csv")
            _log("run_analysis success")
            return (
                _fig_layer_means(stats),
                _fig_token_layer_heatmap(stats),
                _fig_outlier_token_feature_3d(stats),
                _fig_token_layer_maxabs_3d(stats),
                csv_rows,
                csv_cols,
                _fig_feature_magnitudes_per_layer(stats),
                True,
                f"{load_msg}\nAnalysis complete. Stats: {wrapped.dump_stats_path}",
                "Stage 3/3 complete: charts built",
                100,
            )
        except Exception as e:
            _log("run_analysis failed error=%s\n%s", e, traceback.format_exc())
            empty = _empty_fig("Run failed")
            return (
                empty,
                empty,
                empty,
                empty,
                [],
                [],
                [],
                False,
                f"Run failed: {e}",
                "Stage 2/3 failed: inference",
                0,
            )
        finally:
            if wrapped is not None:
                try:
                    wrapped.unregister_hooks()
                except Exception:
                    pass

    return app


def launch(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    app = create_app()
    _log(
        "launch webui host=%s port=%s debug=%s pid=%s device=%s log_file=%s",
        host,
        port,
        debug,
        os.getpid(),
        _preferred_device(),
        _LOG_PATH,
    )
    app.run(host=host, port=port, debug=debug, threaded=False)
