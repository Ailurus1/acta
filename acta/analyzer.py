from __future__ import annotations

import fnmatch
import json
import re
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


class _AnalyzerModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dump_stats_path: str,
        target_layers: str | list[str] | None = None,
        draw_charts: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.dump_stats_path = dump_stats_path
        self.target_layers = target_layers
        self.draw_charts = draw_charts
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._layer_values: dict[str, list[torch.Tensor]] = {}
        self._register_hooks()

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
            # Support shell-style wildcard filters like "*_proj".
            if fnmatch.fnmatchcase(layer_name, pattern) or fnmatch.fnmatchcase(class_name, pattern):
                return True

            # Also support raw regular expressions.
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
        with open(self.dump_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        if self.draw_charts:
            chart_dir = Path(self.dump_stats_path).with_suffix("")
            draw_activation_charts(stats=stats, output_dir=chart_dir.as_posix())

    def _run_and_dump(self, runner: Any, *args: Any, **kwargs: Any) -> Any:
        result = runner(*args, **kwargs)
        self._dump_stats()
        return result

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_and_dump(self.model.forward, *args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.model, "generate"):
            raise AttributeError("Wrapped model has no 'generate' method")
        return self._run_and_dump(self.model.generate, *args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.model, item)


def AutoAnalyzer(
    model: nn.Module,
    dump_stats_path: str = "./activations_analysis.json",
    target_layers: str | list[str] | None = None,
    draw_charts: bool = False,
) -> nn.Module:
    return _AnalyzerModel(
        model=model,
        dump_stats_path=dump_stats_path,
        target_layers=target_layers,
        draw_charts=draw_charts,
    )
