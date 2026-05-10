from __future__ import annotations

import base64
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


def _img_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def _build_3d_gallery(run_dir: Path) -> list[Any]:
    items: list[Any] = []
    main_3d = [
        run_dir / "outlier_token_feature_3d.png",
        run_dir / "token_layer_maxabs_3d.png",
    ]
    for p in main_3d:
        src = _img_data_uri(p)
        if src is None:
            continue
        items.append(
            html.Div(
                [
                    html.P(p.name, className="acta-label"),
                    html.Img(src=src, className="acta-chart-img"),
                ],
                className="acta-chart-card",
            )
        )
    per_layer = run_dir / "outlier_token_feature_3d_per_layer"
    if per_layer.exists():
        shown = 0
        for p in sorted(per_layer.glob("*.png")):
            src = _img_data_uri(p)
            if src is None:
                continue
            items.append(
                html.Div(
                    [
                        html.P(f"Per-layer: {p.stem}", className="acta-label"),
                        html.Img(src=src, className="acta-chart-img"),
                    ],
                    className="acta-chart-card",
                )
            )
            shown += 1
            if shown >= 4:
                break
    if not items:
        return [html.P("No visualizer 3D charts generated for this run.", className="acta-sub")]
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
            if isinstance(m, list) and m:
                names.append(ln)
                vals.append(float(sum(float(x) for x in m) / max(len(m), 1)))
    fig = go.Figure(data=[go.Bar(x=names, y=vals, marker_color="#38bdf8")])
    fig.update_layout(title="Per-layer mean activations", xaxis_title="Layer")
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
            dcc.Store(id="model_loaded", data=False),
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
                                        "Load model",
                                        id="btn_load",
                                        n_clicks=0,
                                        className="acta-btn acta-btn-primary",
                                    ),
                                    html.Span(
                                        [
                                            html.Button(
                                                "Run analysis",
                                                id="btn_run",
                                                n_clicks=0,
                                                className="acta-btn acta-btn-run",
                                            ),
                                        ],
                                        id="run-wrap",
                                        style={"display": "none", "marginLeft": "10px"},
                                    ),
                                ],
                                className="acta-row",
                                style={"marginTop": "14px"},
                            ),
                            html.Div(
                                [
                                    html.P("Progress", className="acta-label"),
                                    html.P(
                                        "Idle",
                                        id="progress-title",
                                        className="acta-sub",
                                        style={"margin": "0 0 6px"},
                                    ),
                                    html.Progress(
                                        id="stage-progress",
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
                    html.Div(
                        [
                            dcc.Graph(id="fig_layer", config={"scrollZoom": True}),
                            dcc.Graph(id="fig_heatmap", config={"scrollZoom": True}),
                            dcc.Graph(id="fig_outliers", config={"scrollZoom": True}),
                            html.H4("CSV Results", className="acta-h4"),
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
                            html.H4("Outlier table", className="acta-h4"),
                            html.Pre(id="table", className="acta-table-wrap"),
                            html.H4("Visualizer 3D charts", className="acta-h4"),
                            html.Div(id="viz-3d", className="acta-chart-grid"),
                        ],
                        className="acta-card",
                    ),
                ],
                className="acta-shell",
            ),
        ]
    )

    @app.callback(
        Output("payload", "value"),
        Input("btn_suggest", "n_clicks"),
        State("source", "value"),
        State("task", "value"),
        prevent_initial_call=True,
    )
    def suggest_payload(_: int, source: str, task: str) -> str:
        return _sample_payload(source, task)

    @app.callback(
        Output("run-wrap", "style"),
        Input("model_loaded", "data"),
    )
    def toggle_run_button(loaded: bool) -> dict[str, Any]:
        if loaded:
            return {
                "display": "inline-block",
                "marginLeft": "10px",
                "verticalAlign": "middle",
            }
        return {"display": "none"}

    @app.callback(
        Output("model_loaded", "data"),
        Output("status", "children"),
        Output("progress-title", "children"),
        Output("stage-progress", "value"),
        Input("btn_load", "n_clicks"),
        State("source", "value"),
        State("task", "value"),
        State("hf_name", "value"),
        State("local_path", "value"),
        running=[
            (Output("progress-title", "children"), "Stage 1/3: Loading model...", "Idle"),
            (Output("stage-progress", "value"), 15, 0),
        ],
        prevent_initial_call=True,
    )
    def load_model(
        _: int, source: str, task: str, hf_name: str, local_path: str
    ) -> tuple[bool, str, str, int]:
        _log("load_model start source=%s task=%s pid=%s", source, task, os.getpid())
        try:
            device = _preferred_device()
            _log("selected device=%s", device)
            if source == "hf":
                name = (hf_name or "").strip()
                if not name:
                    _log("load_model missing hf name")
                    return False, "Please provide a HuggingFace model name.", "Idle", 0
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
                    }
                )
                _log("load_model success hf")
                return True, (
                    f"Loaded HuggingFace model on {device}: {name} ({task})."
                ), "Stage 1/3 complete: Model loaded", 33
            path = Path((local_path or "").strip()).expanduser().resolve()
            if not path.exists():
                _log("load_model local missing path=%s", path)
                return False, f"Local file not found: {path}", "Idle", 0
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
                }
            )
            _log("load_model success local")
            return True, f"Loaded local model on {device} from:\n{path}", "Stage 1/3 complete: Model loaded", 33
        except Exception as e:
            _log("load_model failed error=%s\n%s", e, traceback.format_exc())
            return False, f"Load failed: {e}", "Idle", 0

    @app.callback(
        Output("fig_layer", "figure"),
        Output("fig_heatmap", "figure"),
        Output("fig_outliers", "figure"),
        Output("csv-table", "data"),
        Output("csv-table", "columns"),
        Output("table", "children"),
        Output("viz-3d", "children"),
        Output("status", "children", allow_duplicate=True),
        Output("progress-title", "children", allow_duplicate=True),
        Output("stage-progress", "value", allow_duplicate=True),
        Input("btn_run", "n_clicks"),
        State("payload", "value"),
        State("target_layers", "value"),
        running=[
            (
                Output("progress-title", "children"),
                "Stage 2/3: Running inference...",
                "Stage 1/3 complete: Model loaded",
            ),
            (Output("stage-progress", "value"), 55, 33),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(
        _: int,
        payload_text: str,
        target_layers: str | None,
    ) -> tuple[
        Any,
        Any,
        Any,
        list[dict[str, Any]],
        list[dict[str, str]],
        str,
        list[Any],
        str,
        str,
        int,
    ]:
        model = _STATE.get("model")
        if model is None:
            empty = _empty_fig("No model loaded")
            return empty, empty, empty, [], [], "", [], "Load model first.", "Idle", 0
        tokenizer = _STATE.get("tokenizer")
        source = _STATE.get("source")
        task = _STATE.get("task")
        device = _STATE.get("device", torch.device("cpu"))
        _log(
            "run_analysis start source=%s task=%s model=%s device=%s payload_chars=%d",
            source,
            task,
            model.__class__.__name__,
            device,
            len(payload_text or ""),
        )
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
            table = str(stats.get("outliers", {}).get("table", ""))
            from acta.visualizer import build_charts_from_stats_file

            run_dir = Path(stats.get("_acta", {}).get("output_run_dir", tmp))
            _log("stage 3/3: building charts")
            build_charts_from_stats_file(stats_path=wrapped.dump_stats_path, output_dir=run_dir)
            csv_rows, csv_cols = _csv_for_dash(run_dir / "acta_results.csv")
            viz_children = _build_3d_gallery(run_dir)
            _log("run_analysis success")
            return (
                _fig_layer_means(stats),
                _fig_token_layer_heatmap(stats),
                _fig_outlier_flags(stats),
                csv_rows,
                csv_cols,
                table,
                viz_children,
                f"Analysis complete. Stats: {wrapped.dump_stats_path}",
                "Stage 3/3 complete: Charts built",
                100,
            )
        except Exception as e:
            _log("run_analysis failed error=%s\n%s", e, traceback.format_exc())
            empty = _empty_fig("Run failed")
            return empty, empty, empty, [], [], "", [], f"Run failed: {e}", "Idle", 0

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
