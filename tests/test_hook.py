from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from acta import AutoAnalyzer


def test_apply_hook_to_model_topology(tmp_path: Path) -> None:
    base_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    model = AutoAnalyzer(
        base_model,
        dump_stats_path=str(tmp_path / "hook_topology"),
        target_layers="*Linear*",
        draw_charts=False,
        verbose=False,
    )

    linear_modules = [m for m in model.model.modules() if isinstance(m, nn.Linear)]
    assert linear_modules, "Expected at least one linear module"
    assert all(
        len(m._forward_hooks) > 0 for m in linear_modules
    ), "Expected hooks on targeted modules"


def test_hook_creates_required_output_files(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    model = AutoAnalyzer(
        base_model,
        dump_stats_path=str(tmp_path / "hook_outputs"),
        target_layers="*",
        draw_charts=False,
        verbose=False,
    )
    model.eval()

    with torch.no_grad():
        _ = model(torch.randn(2, 8))

    stats_path = Path(model.dump_stats_path)
    run_dir = Path(model.output_run_dir)
    csv_path = run_dir / "acta_results.csv"

    assert run_dir.exists()
    assert stats_path.exists()
    assert csv_path.exists()

    data = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "layers" in data
    assert "_acta" in data
