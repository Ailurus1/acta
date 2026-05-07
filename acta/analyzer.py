from __future__ import annotations

import csv
import fnmatch
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .visualizer import draw_activation_charts


def _to_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _channel_stats(values: torch.Tensor) -> dict[str, list[float]]:
    if values.numel() == 0:
        return {
            "mean": [],
            "variance": [],
            "quantiles": {"q25": [], "q50": [], "q75": []},
            "kurtosis": [],
        }

    mean = values.mean(dim=0)
    var = values.var(dim=0, unbiased=False)
    q = torch.quantile(values, q=torch.tensor([0.25, 0.5, 0.75], device=values.device), dim=0)

    centered = values - mean
    m2 = torch.mean(centered.pow(2), dim=0)
    m4 = torch.mean(centered.pow(4), dim=0)
    eps = torch.finfo(values.dtype).eps
    kurtosis = m4 / (m2.pow(2) + eps)

    return {
        "mean": mean.cpu().tolist(),
        "variance": var.cpu().tolist(),
        "quantiles": {
            "q25": q[0].cpu().tolist(),
            "q50": q[1].cpu().tolist(),
            "q75": q[2].cpu().tolist(),
        },
        "kurtosis": kurtosis.cpu().tolist(),
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
    outliers: list[bool],
    layers: list[str | None],
    channel_dims: list[int | None],
) -> str:
    def _short(s: str, max_len: int = 20) -> str:
        s = s.replace("\n", "\\n")
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    lines = []
    header = (
        f"{'idx':>5}  {'token':<22}  {'token_id':>10}  {'layer':<32}  {'channel_dim':>11}"
        f"  {'outliers':>15}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for i, (tok, tid, layer, dim, o) in enumerate(
        zip(tokens, token_ids, layers, channel_dims, outliers, strict=False)
    ):
        layer_s = "" if layer is None else layer
        dim_s = "" if dim is None else str(dim)
        lines.append(
            f"{i:>5}  {_short(tok):<22}  {tid:>10}  {layer_s:<32}  {dim_s:>11}  {str(o):>15}"
        )
    return "\n".join(lines)


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
        outlier_threshold: float = 6.0,
        hard_layers_frac: float = 0.25,
        hard_seqdim_frac: float = 0.06,
        verbose: bool = False,
        tokenizer: Any | None = None,
        vit_reg_patch_labels: bool = False,
        asr_chunk_labels: bool = False,
        chart_3d_max_tokens: int = 24,
    ) -> None:
        super().__init__()
        self.model = model
        self._dump_stats_path_spec = dump_stats_path
        self._session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dump_stats_root, self.output_run_dir, self.dump_stats_path = self._resolve_versioned_dump_paths()
        self.target_layers = target_layers
        self.draw_charts = draw_charts
        self.outlier_threshold = float(outlier_threshold)
        self.hard_layers_frac = float(hard_layers_frac)
        self.hard_seqdim_frac = float(hard_seqdim_frac)
        self.verbose = bool(verbose)
        self.tokenizer = tokenizer
        self.vit_reg_patch_labels = bool(vit_reg_patch_labels)
        self.asr_chunk_labels = bool(asr_chunk_labels)
        self.chart_3d_max_tokens = int(chart_3d_max_tokens)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._layer_values: dict[str, list[torch.Tensor]] = {}
        self._in_generate: bool = False
        self._run_token_frac_by_layer: dict[str, list[torch.Tensor]] = {}
        self._run_token_mean_by_layer: dict[str, list[torch.Tensor]] = {}
        self._run_token_var_by_layer: dict[str, list[torch.Tensor]] = {}
        self._last_prompt_token_ids: list[int] | None = None
        self._prompt_len: int | None = None
        self._run_feature_token_count_by_layer: dict[str, int] = {}
        self._run_feature_counts_by_layer: dict[str, torch.Tensor] = {}
        self._run_max_abs_token_feature_by_layer: dict[str, torch.Tensor] = {}  # layer -> [T_prompt, H]
        self._run_max_abs_token_feature: torch.Tensor | None = None  # [T_prompt, H]
        self._run_hidden_dim: int | None = None
        self._run_argmax_layer_per_token_feature: torch.Tensor | None = None  # [T_prompt, H] int32
        self._run_layer_order: list[str] = []
        self._run_layer_to_index: dict[str, int] = {}
        self._run_token_maxabs_by_layer: dict[str, list[torch.Tensor]] = {}
        self._last_generate_outliers: dict[str, Any] | None = None
        self._register_hooks()

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
        fieldnames = ["idx", "token", "token_id", "layer", "channel_dim", "outliers"]
        rows: list[dict[str, Any]] = []
        if self._last_generate_outliers is not None:
            tokens = self._last_generate_outliers.get("prompt_tokens", [])
            tids = self._last_generate_outliers.get("prompt_token_ids", [])
            outlier_flags = self._last_generate_outliers.get("outliers", [])
            layers = self._last_generate_outliers.get("per_token_layer", [])
            dims = self._last_generate_outliers.get("per_token_channel_dim", [])
            for i, tok in enumerate(tokens):
                rows.append(
                    {
                        "idx": i,
                        "token": tok,
                        "token_id": tids[i] if i < len(tids) else "",
                        "layer": layers[i] if i < len(layers) and layers[i] is not None else "",
                        "channel_dim": dims[i] if i < len(dims) and dims[i] is not None else "",
                        "outliers": outlier_flags[i] if i < len(outlier_flags) else "",
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

    def _maybe_set_prompt_len_from_tensor(self, tensor: torch.Tensor) -> None:
        if self._prompt_len is None and tensor.ndim == 3:
            # Works for LLMs and ViTs: treat dim=1 as sequence length.
            self._prompt_len = int(tensor.shape[1])

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

        patterns = self.target_layers if isinstance(self.target_layers, list) else [self.target_layers]
        class_name = module.__class__.__name__
        for pattern in patterns:
            if fnmatch.fnmatchcase(layer_name, pattern) or fnmatch.fnmatchcase(class_name, pattern):
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

            def _hook_fn(mod: nn.Module, inputs: tuple[Any, ...], output: Any, layer_name: str = name) -> None:
                tensor = _to_tensor(output)
                if tensor is None:
                    return
                if not tensor.is_floating_point():
                    return

                if self._in_generate:
                    self._maybe_set_prompt_len_from_tensor(tensor)
                    frac = _token_outlier_fraction(tensor, threshold=self.outlier_threshold)
                    if frac is not None:
                        self._run_token_frac_by_layer.setdefault(layer_name, []).append(frac.cpu())
                        token_mean = tensor.detach().to(torch.float32).mean(dim=-1)  # [B, S]
                        token_var = tensor.detach().to(torch.float32).var(dim=-1, unbiased=False)  # [B, S]
                        self._run_token_mean_by_layer.setdefault(layer_name, []).append(token_mean.cpu())
                        self._run_token_var_by_layer.setdefault(layer_name, []).append(token_var.cpu())

                        if tensor.ndim == 3 and self._prompt_len is not None:
                            b, s, h = tensor.shape
                            t = min(int(self._prompt_len), int(s))
                            if t > 0:
                                abs_act = tensor.detach().abs().to(torch.float32)[0, :t, :]  # [T, H]

                                tok_max = abs_act.max(dim=-1).values.cpu()  # [T]
                                self._run_token_maxabs_by_layer.setdefault(layer_name, []).append(tok_max)

                                # Per-layer per-token/per-feature max magnitude across generation steps.
                                abs_cpu = abs_act.cpu()
                                if layer_name not in self._run_max_abs_token_feature_by_layer:
                                    self._run_max_abs_token_feature_by_layer[layer_name] = abs_cpu
                                else:
                                    prev_layer = self._run_max_abs_token_feature_by_layer[layer_name]
                                    if prev_layer.shape == abs_cpu.shape:
                                        self._run_max_abs_token_feature_by_layer[layer_name] = torch.maximum(prev_layer, abs_cpu)

                                if self._run_max_abs_token_feature is None or self._run_hidden_dim != int(h):
                                    self._run_hidden_dim = int(h)
                                    self._run_max_abs_token_feature = abs_act.cpu()
                                    layer_idx = self._run_layer_to_index.setdefault(layer_name, len(self._run_layer_order))
                                    if layer_idx == len(self._run_layer_order):
                                        self._run_layer_order.append(layer_name)
                                    self._run_argmax_layer_per_token_feature = torch.full(
                                        (t, int(h)),
                                        int(layer_idx),
                                        dtype=torch.int32,
                                    )
                                else:
                                    layer_idx = self._run_layer_to_index.setdefault(layer_name, len(self._run_layer_order))
                                    if layer_idx == len(self._run_layer_order):
                                        self._run_layer_order.append(layer_name)
                                    prev_max = self._run_max_abs_token_feature
                                    cur = abs_act.cpu()
                                    updated_max = torch.maximum(prev_max, cur)
                                    self._run_max_abs_token_feature = updated_max
                                    if self._run_argmax_layer_per_token_feature is not None:
                                        upd = cur > prev_max
                                        self._run_argmax_layer_per_token_feature[upd] = int(layer_idx)

                                mask = abs_act >= float(self.outlier_threshold)  # [T, H]
                                self._run_feature_token_count_by_layer[layer_name] = (
                                    self._run_feature_token_count_by_layer.get(layer_name, 0) + t
                                )
                                if layer_name not in self._run_feature_counts_by_layer:
                                    self._run_feature_counts_by_layer[layer_name] = mask.sum(dim=0).cpu()
                                else:
                                    self._run_feature_counts_by_layer[layer_name] += mask.sum(dim=0).cpu()

                flattened = _prepare_per_channel(tensor, mod)
                self._layer_values.setdefault(layer_name, []).append(flattened.cpu())

            self._hooks.append(module.register_forward_hook(_hook_fn))

    def reset_stats(self) -> None:
        self._layer_values.clear()

    def _build_stats(self) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for layer_name, chunks in self._layer_values.items():
            if not chunks:
                continue
            data = torch.cat(chunks, dim=0)
            layer_stats = _channel_stats(data)
            layer_stats["num_observations"] = data.shape[0]
            layers[layer_name] = layer_stats
        return {"layers": layers}

    def _dump_stats(self) -> None:
        stats = self._build_stats()
        if self._last_generate_outliers is not None:
            self._last_generate_outliers["chart_3d_max_tokens"] = self.chart_3d_max_tokens
            stats["outliers"] = self._last_generate_outliers
        stats["_acta"] = {
            "dump_stats_root": self.dump_stats_root.as_posix(),
            "output_run_dir": self.output_run_dir.as_posix(),
            "dump_stats_path": self.dump_stats_path,
            "session_timestamp": self._session_timestamp,
            "results_csv": (self.dump_stats_root / "acta_results.csv").as_posix(),
        }
        with open(self.dump_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        self._write_acta_results_csv()
        if self.draw_charts:
            chart_dir = Path(self.dump_stats_path).parent
            draw_activation_charts(stats=stats, output_dir=chart_dir.as_posix())

    def _run_and_dump(self, runner: Any, *args: Any, **kwargs: Any) -> Any:
        result = runner(*args, **kwargs)
        self._dump_stats()
        return result

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._reset_run_sequence_buffers()
        self._in_generate = True
        try:
            result = self.model.forward(*args, **kwargs)
        finally:
            self._in_generate = False

        # Build a pseudo "outliers" payload based on collected sequence stats, if any.
        if self._prompt_len is not None:
            use_len = int(self._prompt_len)
            token_ids = list(range(use_len))
            decoded = (
                self._sequence_position_labels(use_len)
                if self.tokenizer is None
                else [decode_token(self.tokenizer, i) for i in token_ids]
            )
            fake = torch.tensor([token_ids], dtype=torch.long)
            self._last_generate_outliers = self._detect_outliers_after_generate(fake, prompt_len=use_len)
            if self._last_generate_outliers is not None:
                self._last_generate_outliers["prompt_token_ids"] = token_ids
                self._last_generate_outliers["prompt_tokens"] = decoded
                if "token_trends" in self._last_generate_outliers and isinstance(self._last_generate_outliers["token_trends"], dict):
                    self._last_generate_outliers["token_trends"]["tokens"] = decoded
                    self._last_generate_outliers["token_trends"]["token_ids"] = token_ids
                if "outliers" in self._last_generate_outliers:
                    outlier_flags = self._last_generate_outliers.get("outliers", [])
                    layers = self._last_generate_outliers.get("per_token_layer", [None] * len(decoded))
                    dims = self._last_generate_outliers.get("per_token_channel_dim", [None] * len(decoded))
                    self._last_generate_outliers["table"] = _format_outlier_table_decoded(
                        tokens=decoded,
                        token_ids=token_ids,
                        outliers=outlier_flags,
                        layers=layers,
                        channel_dims=dims,
                    )

            if self.verbose and self._last_generate_outliers is not None:
                table = self._last_generate_outliers.get("table")
                if table:
                    print(table)

        self._dump_stats()
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

        if isinstance(result, torch.Tensor) and result.ndim == 2 and prompt_len is not None:
            self._last_prompt_token_ids = result[0, :prompt_len].detach().cpu().tolist()

        self._last_generate_outliers = self._detect_outliers_after_generate(result, prompt_len=prompt_len)
        if self.verbose and self._last_generate_outliers is not None:
            table = self._last_generate_outliers.get("table")
            if table:
                print(table)
        self._dump_stats()
        return result

    def _detect_outliers_after_generate(self, generate_result: Any, prompt_len: int | None) -> dict[str, Any] | None:
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

        t_min = min(t.shape[1] for t in per_layer_frac.values())
        frac_stack = torch.stack([per_layer_frac[name][:, :t_min] for name in layer_names], dim=0)  # [L, B, T]

        # Hard (LLM.int8-style): >=25% layers have >=seqdim_fraction hidden dims exceeding threshold.
        layer_affected = frac_stack >= self.hard_seqdim_frac  # [L, B, T]
        frac_layers_affected = layer_affected.to(torch.float32).mean(dim=0)  # [B, T]
        hard_mask = frac_layers_affected >= self.hard_layers_frac

        token_ids_full = generate_result[0, :t_min].detach().cpu().tolist()
        hard_full = hard_mask[0].detach().cpu().tolist()

        use_len = min(prompt_len, t_min) if prompt_len is not None else t_min
        token_ids = token_ids_full[:use_len]
        outliers_list = hard_full[:use_len]
        decoded_tokens = [decode_token(self.tokenizer, tid) for tid in token_ids]

        per_layer_mean: dict[str, torch.Tensor] = {}
        per_layer_var: dict[str, torch.Tensor] = {}
        for layer_name in layer_names:
            mean_chunks = self._run_token_mean_by_layer.get(layer_name, [])
            var_chunks = self._run_token_var_by_layer.get(layer_name, [])
            if not mean_chunks or not var_chunks:
                continue
            per_layer_mean[layer_name] = torch.cat(mean_chunks, dim=1)[:, :t_min]  # [B, T]
            per_layer_var[layer_name] = torch.cat(var_chunks, dim=1)[:, :t_min]  # [B, T]

        token_trends: dict[str, Any] | None = None
        if len(per_layer_mean) == len(layer_names):
            means_layer_token = []
            vars_layer_token = []
            for ln in layer_names:
                means_layer_token.append(per_layer_mean[ln][0, :use_len].cpu().tolist())
                vars_layer_token.append(per_layer_var[ln][0, :use_len].cpu().tolist())
            token_trends = {
                "layer_names": layer_names,
                "token_count": int(use_len),
                # shape: [num_layers][token_count]
                "mean": means_layer_token,
                "variance": vars_layer_token,
                "tokens": decoded_tokens,
                "token_ids": token_ids,
            }

        # Feature-dimension outlier detection (LLM.int8-style) and 3D plot payload.
        outlier_feature_dims: list[int] = []
        h_shared: int | None = None
        if self._run_feature_counts_by_layer and self._run_feature_token_count_by_layer:
            eligible_layers = [ln for ln in layer_names if ln in self._run_feature_counts_by_layer]
            if eligible_layers:
                # Only compare layers that share the same feature dimension H.
                lengths = [int(self._run_feature_counts_by_layer[ln].shape[0]) for ln in eligible_layers]
                h = int(max(set(lengths), key=lengths.count))
                eligible_layers = [ln for ln in eligible_layers if int(self._run_feature_counts_by_layer[ln].shape[0]) == h]
                h_shared = h
                affected_layers_per_dim = torch.zeros(h, dtype=torch.float32)
                num_layers = len(eligible_layers)
                if num_layers > 0:
                    for ln in eligible_layers:
                        total_toks = max(int(self._run_feature_token_count_by_layer.get(ln, 0)), 1)
                        frac_tokens = self._run_feature_counts_by_layer[ln].to(torch.float32) / float(total_toks)
                        affected_layers_per_dim += (frac_tokens >= float(self.hard_seqdim_frac)).to(torch.float32)
                    frac_layers = affected_layers_per_dim / float(num_layers)
                    outlier_feature_dims = (
                        (frac_layers >= float(self.hard_layers_frac)).nonzero(as_tuple=False).view(-1).cpu().tolist()
                    )

        token_feature_magnitude: list[list[float]] | None = None
        if self._run_max_abs_token_feature is not None and outlier_feature_dims:
            max_abs = self._run_max_abs_token_feature[:use_len, :]  # [T, H]
            token_feature_magnitude = max_abs[:, outlier_feature_dims].cpu().tolist()  # [T][F]

        token_feature_magnitude_by_layer: dict[str, list[list[float]]] = {}
        h_for_slices = h_shared
        if h_for_slices is None and self._run_max_abs_token_feature is not None:
            h_for_slices = int(self._run_max_abs_token_feature.shape[1])
        if outlier_feature_dims and self._run_max_abs_token_feature_by_layer and h_for_slices is not None:
            for ln, ten in self._run_max_abs_token_feature_by_layer.items():
                if ten.ndim != 2:
                    continue
                if int(ten.shape[1]) != int(h_for_slices):
                    continue
                t_use = min(int(use_len), int(ten.shape[0]))
                if t_use <= 0:
                    continue
                token_feature_magnitude_by_layer[ln] = ten[:t_use, outlier_feature_dims].cpu().tolist()

        # For each token: pick the strongest outlier dim (among detected outlier dims).
        per_token_layer: list[str | None] = [None] * int(use_len)
        per_token_channel_dim: list[int | None] = [None] * int(use_len)
        if self._run_max_abs_token_feature is not None and outlier_feature_dims:
            max_abs = self._run_max_abs_token_feature[:use_len, :]  # [T, H]
            for ti in range(int(use_len)):
                vals = max_abs[ti, outlier_feature_dims]
                if vals.numel() == 0:
                    continue
                j = int(torch.argmax(vals).item())
                per_token_channel_dim[ti] = int(outlier_feature_dims[j])
                if self._run_argmax_layer_per_token_feature is not None and self._run_layer_order:
                    li = int(self._run_argmax_layer_per_token_feature[ti, per_token_channel_dim[ti]].item())
                    if 0 <= li < len(self._run_layer_order):
                        per_token_layer[ti] = self._run_layer_order[li]

        # 3D chart payload for all layers: token × layer × max-magnitude (over hidden dims).
        token_layer_max_magnitude: list[list[float]] | None = None
        if self._run_token_maxabs_by_layer:
            layer_list = sorted(self._run_token_maxabs_by_layer.keys())
            t_layer_min = min(torch.cat(chunks, dim=0).shape[0] for chunks in self._run_token_maxabs_by_layer.values())
            t_layer_use = min(int(use_len), int(t_layer_min))
            token_layer_max_magnitude = []
            for ln in layer_list:
                seq = torch.cat(self._run_token_maxabs_by_layer[ln], dim=0)[:t_layer_use]  # [T]
                token_layer_max_magnitude.append(seq.cpu().tolist())

        return {
            "threshold": self.outlier_threshold,
            "hard_criteria": {
                "layers_fraction_threshold": self.hard_layers_frac,
                "seqdim_fraction_threshold": self.hard_seqdim_frac,
            },
            "num_layers_considered": len(layer_names),
            "token_count": int(use_len),
            "outliers": outliers_list,
            "prompt_token_ids": token_ids,
            "prompt_tokens": decoded_tokens,
            "per_token_layer": per_token_layer,
            "per_token_channel_dim": per_token_channel_dim,
            "table": _format_outlier_table_decoded(
                tokens=decoded_tokens,
                token_ids=token_ids,
                outliers=outliers_list,
                layers=per_token_layer,
                channel_dims=per_token_channel_dim,
            ),
            "token_trends": token_trends,
            "outlier_feature_dims": outlier_feature_dims,
            "token_feature_magnitude": token_feature_magnitude,  # [token][feature] magnitudes (abs max over layers)
            "token_feature_magnitude_by_layer": token_feature_magnitude_by_layer,
            "token_layer_trends": {
                "layer_names": sorted(self._run_token_maxabs_by_layer.keys()),
                "token_count": int(use_len),
                "tokens": decoded_tokens,
                "token_ids": token_ids,
                # shape: [num_layers][token_count]
                "max_abs": token_layer_max_magnitude,
            }
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
    outlier_threshold: float = 6.0,
    hard_layers_frac: float = 0.25,
    hard_seqdim_frac: float = 0.06,
    verbose: bool = False,
    tokenizer: Any | None = None,
    vit_reg_patch_labels: bool = False,
    asr_chunk_labels: bool = False,
    chart_3d_max_tokens: int = 24,
) -> nn.Module:
    return _AnalyzerModel(
        model=model,
        dump_stats_path=dump_stats_path,
        target_layers=target_layers,
        draw_charts=draw_charts,
        outlier_threshold=outlier_threshold,
        hard_layers_frac=hard_layers_frac,
        hard_seqdim_frac=hard_seqdim_frac,
        verbose=verbose,
        tokenizer=tokenizer,
        vit_reg_patch_labels=vit_reg_patch_labels,
        asr_chunk_labels=asr_chunk_labels,
        chart_3d_max_tokens=chart_3d_max_tokens,
    )
