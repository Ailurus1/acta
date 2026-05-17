from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch import nn
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    GPT2Config,
    GPT2LMHeadModel,
    ViTConfig,
    ViTForImageClassification,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from acta import AutoAnalyzer
from acta.analyzer import _IQR_FLAG, _SOFT_FLAG, _sync_outlier_attribution_columns

ModelBuilder = Callable[[], nn.Module]
InputBuilder = Callable[[], dict[str, Any]]
RunFn = Callable[[nn.Module, dict[str, Any]], Any]


def _gpt2_builder() -> nn.Module:
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        bos_token_id=1,
        eos_token_id=2,
    )
    return GPT2LMHeadModel(cfg).eval()


def _gpt2_input() -> dict[str, Any]:
    return {"input_ids": torch.tensor([[1, 5, 7, 9, 11, 2]], dtype=torch.long)}


def _gpt2_batched_input() -> dict[str, Any]:
    return {
        "input_ids": torch.tensor(
            [
                [1, 5, 7, 9, 11, 2],
                [1, 6, 8, 10, 12, 2],
            ],
            dtype=torch.long,
        )
    }


def _gpt2_run(model: nn.Module, inp: dict[str, Any]) -> Any:
    with torch.no_grad():
        return model.generate(**inp, max_new_tokens=3)


def _distilbert_builder() -> nn.Module:
    cfg = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=64,
        n_layers=2,
        n_heads=4,
        dim=32,
        hidden_dim=64,
    )
    return DistilBertModel(cfg).eval()


def _distilbert_input() -> dict[str, Any]:
    input_ids = torch.tensor(
        [[3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]], dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones((1, 12), dtype=torch.long),
    }


def _distilbert_batched_input() -> dict[str, Any]:
    input_ids = torch.tensor(
        [
            [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        ],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones((2, 12), dtype=torch.long),
    }


def _forward_run(model: nn.Module, inp: dict[str, Any]) -> Any:
    with torch.no_grad():
        return model(**inp)


def _vit_builder() -> nn.Module:
    cfg = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=2,
    )
    return ViTForImageClassification(cfg).eval()


def _vit_input() -> dict[str, Any]:
    vals = torch.linspace(
        -1.0, 1.0, steps=1 * 3 * 32 * 32, dtype=torch.float32
    ).reshape(1, 3, 32, 32)
    return {"pixel_values": vals}


def _vit_batched_input() -> dict[str, Any]:
    vals = torch.linspace(
        -1.0, 1.0, steps=2 * 3 * 32 * 32, dtype=torch.float32
    ).reshape(2, 3, 32, 32)
    return {"pixel_values": vals}


def _whisper_builder() -> nn.Module:
    cfg = WhisperConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=64,
        max_target_positions=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        decoder_start_token_id=1,
    )
    return WhisperForConditionalGeneration(cfg).get_encoder().eval()


def _whisper_input() -> dict[str, Any]:
    vals = torch.linspace(-0.5, 0.5, steps=1 * 80 * 128, dtype=torch.float32).reshape(
        1, 80, 128
    )
    return {"input_features": vals}


def _whisper_batched_input() -> dict[str, Any]:
    vals = torch.linspace(-0.5, 0.5, steps=2 * 80 * 128, dtype=torch.float32).reshape(
        2, 80, 128
    )
    return {"input_features": vals}


CASES: list[tuple[str, ModelBuilder, InputBuilder, RunFn]] = [
    ("gpt2", _gpt2_builder, _gpt2_input, _gpt2_run),
    ("whisper_tiny", _whisper_builder, _whisper_input, _forward_run),
    ("distilbert", _distilbert_builder, _distilbert_input, _forward_run),
    ("vit", _vit_builder, _vit_input, _forward_run),
]

BATCH_CASES: list[tuple[str, ModelBuilder, InputBuilder, RunFn]] = [
    ("gpt2", _gpt2_builder, _gpt2_batched_input, _gpt2_run),
    ("whisper_tiny", _whisper_builder, _whisper_batched_input, _forward_run),
    ("distilbert", _distilbert_builder, _distilbert_batched_input, _forward_run),
    ("vit", _vit_builder, _vit_batched_input, _forward_run),
]


def _read_stats(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_case(
    tmp_path: Path,
    case_name: str,
    model_builder: ModelBuilder,
    input_builder: InputBuilder,
    run_fn: RunFn,
    *,
    draw_charts: bool,
    num_calls: int = 1,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(42)
    wrapped = AutoAnalyzer(
        model_builder(),
        dump_stats_path=str(tmp_path / case_name),
        target_layers=None,
        draw_charts=draw_charts,
        verbose=False,
        tokenizer=None,
        outlier_threshold=0.0,
    )
    wrapped.eval()
    for _ in range(num_calls):
        run_fn(wrapped, input_builder())
    return wrapped, _read_stats(Path(wrapped.dump_stats_path))


@pytest.mark.parametrize("case_name,model_builder,input_builder,run_fn", CASES)
def test_eval_models_create_required_artifacts_no_charts(
    tmp_path: Path,
    case_name: str,
    model_builder: ModelBuilder,
    input_builder: InputBuilder,
    run_fn: RunFn,
) -> None:
    wrapped, stats = _run_case(
        tmp_path,
        case_name,
        model_builder,
        input_builder,
        run_fn,
        draw_charts=False,
        num_calls=1,
    )
    run_dir = Path(wrapped.output_run_dir)

    assert run_dir.exists()
    assert Path(wrapped.dump_stats_path).exists()
    assert (run_dir / "acta_results.csv").exists()
    assert "layers" in stats and stats["layers"]
    assert not list(run_dir.rglob("*.png"))


@pytest.mark.parametrize("case_name,model_builder,input_builder,run_fn", CASES)
def test_multi_call_aggregated_stats_are_correct(
    tmp_path: Path,
    case_name: str,
    model_builder: ModelBuilder,
    input_builder: InputBuilder,
    run_fn: RunFn,
) -> None:
    base_a, stats_a = _run_case(
        tmp_path / "a",
        case_name,
        model_builder,
        input_builder,
        run_fn,
        draw_charts=False,
        num_calls=1,
    )
    _ = base_a
    base_b, stats_b = _run_case(
        tmp_path / "b",
        case_name,
        model_builder,
        input_builder,
        run_fn,
        draw_charts=False,
        num_calls=1,
    )
    _ = base_b
    combined, stats_c = _run_case(
        tmp_path / "combined",
        case_name,
        model_builder,
        input_builder,
        run_fn,
        draw_charts=False,
        num_calls=2,
    )
    _ = combined

    common_layers = sorted(
        set(stats_a["layers"]) & set(stats_b["layers"]) & set(stats_c["layers"])
    )
    assert common_layers, "No common layers found to validate aggregation"
    ln = common_layers[0]

    n1 = int(stats_a["layers"][ln]["num_observations"])
    n2 = int(stats_b["layers"][ln]["num_observations"])
    n3 = int(stats_c["layers"][ln]["num_observations"])
    assert n3 == n1 + n2

    m1 = torch.tensor(stats_a["layers"][ln]["mean"], dtype=torch.float32)
    m2 = torch.tensor(stats_b["layers"][ln]["mean"], dtype=torch.float32)
    m3 = torch.tensor(stats_c["layers"][ln]["mean"], dtype=torch.float32)
    expected = (m1 * n1 + m2 * n2) / float(n1 + n2)
    assert torch.allclose(m3, expected, atol=1e-5, rtol=1e-4)

    out_a = [bool(v) for v in stats_a.get("outliers", {}).get("outliers", [])]
    out_b = [bool(v) for v in stats_b.get("outliers", {}).get("outliers", [])]
    out_c = [bool(v) for v in stats_c.get("outliers", {}).get("outliers", [])]
    if out_c:
        max_len = max(len(out_a), len(out_b), len(out_c))
        out_a += [False] * (max_len - len(out_a))
        out_b += [False] * (max_len - len(out_b))
        out_c += [False] * (max_len - len(out_c))
        expected_or = [a or b for a, b in zip(out_a, out_b, strict=False)]
        assert out_c == expected_or


@pytest.mark.parametrize("case_name,model_builder,input_builder,run_fn", CASES)
def test_eval_models_create_all_charts(
    tmp_path: Path,
    case_name: str,
    model_builder: ModelBuilder,
    input_builder: InputBuilder,
    run_fn: RunFn,
) -> None:
    wrapped, _stats = _run_case(
        tmp_path,
        f"{case_name}_charts",
        model_builder,
        input_builder,
        run_fn,
        draw_charts=True,
        num_calls=1,
    )
    wrapped._finalize_on_exit()
    run_dir = Path(wrapped.output_run_dir)

    assert not (run_dir / "layer_channel_hist").exists()
    assert not (run_dir / "token_trends").exists()
    op_dir = run_dir / "operator_activation_tops"
    assert op_dir.exists()
    assert list(op_dir.glob("*.png"))
    assert (run_dir / "operator_max_kurtosis_across_blocks.png").exists()
    assert (run_dir / "operator_max_median_ratio_across_blocks.png").exists()
    assert (run_dir / "outlier_token_feature_3d.png").exists()
    maxabs_dir = run_dir / "token_layer_maxabs_3d"
    assert maxabs_dir.is_dir()
    assert list(maxabs_dir.glob("*.png"))
    assert not (run_dir / "token_layer_maxabs_3d.png").exists()
    assert (run_dir / "outlier_token_feature_3d_per_layer").exists()
    assert list((run_dir / "outlier_token_feature_3d_per_layer").glob("*.png"))


def test_draw_token_trends_flag_creates_token_trend_charts(tmp_path: Path) -> None:
    torch.manual_seed(42)
    wrapped = AutoAnalyzer(
        _gpt2_builder(),
        dump_stats_path=str(tmp_path / "token_trends_on"),
        target_layers=None,
        draw_charts=True,
        draw_token_trends=True,
        verbose=False,
        tokenizer=None,
        outlier_threshold=0.0,
    )
    wrapped.eval()
    _gpt2_run(wrapped, _gpt2_input())
    wrapped._finalize_on_exit()
    run_dir = Path(wrapped.output_run_dir)
    assert (run_dir / "token_trends").exists()
    assert list((run_dir / "token_trends").glob("*.png"))


def test_finalize_prints_outlier_table_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrapped = AutoAnalyzer(
        _gpt2_builder(),
        dump_stats_path=str(tmp_path / "stdout_table"),
        target_layers=None,
        draw_charts=False,
        verbose=False,
        tokenizer=None,
        outlier_threshold=0.0,
    )
    wrapped.eval()
    _gpt2_run(wrapped, _gpt2_input())
    wrapped._finalize_on_exit()
    captured = capsys.readouterr().out
    assert "idx" in captured and "token" in captured and "outliers detected" in captured
    assert "llm.int8() outliers soft difinition" not in captured
    assert "per_token_max_kurtosis" not in captured


@pytest.mark.parametrize("case_name,model_builder,input_builder,run_fn", BATCH_CASES)
def test_batched_inference_creates_valid_stats_and_outliers(
    tmp_path: Path,
    case_name: str,
    model_builder: ModelBuilder,
    input_builder: InputBuilder,
    run_fn: RunFn,
) -> None:
    wrapped, stats = _run_case(
        tmp_path,
        f"{case_name}_batched",
        model_builder,
        input_builder,
        run_fn,
        draw_charts=False,
        num_calls=1,
    )
    run_dir = Path(wrapped.output_run_dir)
    csv_path = run_dir / "acta_results.csv"

    assert Path(wrapped.dump_stats_path).exists()
    assert csv_path.exists()
    assert "layers" in stats and stats["layers"]
    first_layer = next(iter(stats["layers"].values()))
    assert int(first_layer["num_observations"]) > 0

    outliers = stats.get("outliers", {})
    if isinstance(outliers, dict) and "outliers" in outliers:
        token_count = int(outliers.get("token_count", 0))
        flags = outliers.get("outliers", [])
        assert len(flags) == token_count


def test_sync_outlier_attribution_columns_clears_when_flag_false() -> None:
    payload = {
        "outliers": [False, True, False],
        "per_token_layer": ["bad.layer", "good.layer", "also.bad"],
        "per_token_channel_dim": [1, 7, 9],
    }
    _sync_outlier_attribution_columns(payload)
    assert payload["per_token_layer"][0] is None
    assert payload["per_token_channel_dim"][0] is None
    assert payload["per_token_layer"][1] == "good.layer"
    assert payload["per_token_channel_dim"][1] == 7
    assert payload["per_token_layer"][2] is None
    assert payload["per_token_channel_dim"][2] is None


def test_merge_outliers_drops_stale_layer_when_flags_false(tmp_path: Path) -> None:
    base = nn.Linear(4, 4)
    wrapped = AutoAnalyzer(
        base,
        dump_stats_path=str(tmp_path / "merge_sync"),
        verbose=False,
        target_layers="*",
    )
    wrapped._aggregate_generate_outliers = {
        "outliers": [False],
        "token_count": 1,
        "prompt_tokens": ["x"],
        "prompt_token_ids": [0],
        "per_token_layer": ["stale.layer"],
        "per_token_channel_dim": [99],
    }
    wrapped._merge_outliers_across_calls(
        {
            "outliers": [False],
            "token_count": 1,
            "prompt_tokens": ["x"],
            "prompt_token_ids": [0],
            "per_token_layer": [None],
            "per_token_channel_dim": [None],
        }
    )
    agg = wrapped._aggregate_generate_outliers
    assert agg is not None
    assert agg["outliers"] == [False]
    assert agg["per_token_layer"] == [None]
    assert agg["per_token_channel_dim"] == [None]


def test_stats_outlier_columns_consistent_after_generate(tmp_path: Path) -> None:
    wrapped, stats = _run_case(
        tmp_path,
        "consistency",
        _gpt2_builder,
        _gpt2_input,
        _gpt2_run,
        draw_charts=False,
        num_calls=1,
    )
    _ = wrapped
    out = stats.get("outliers", {})
    if not isinstance(out, dict):
        return
    flags = out.get("outliers", [])
    layers = out.get("per_token_layer", [])
    dims = out.get("per_token_channel_dim", [])
    if not flags:
        return
    n = min(len(flags), len(layers), len(dims))
    for i in range(n):
        if not bool(flags[i]):
            assert layers[i] in (None, "")
            assert dims[i] in (None, "")


def test_stats_include_soft_hard_iqr_columns_and_activation_report(
    tmp_path: Path,
) -> None:
    wrapped, stats = _run_case(
        tmp_path,
        "new_metrics",
        _gpt2_builder,
        _gpt2_input,
        _gpt2_run,
        draw_charts=False,
        num_calls=1,
    )
    _ = wrapped
    out = stats.get("outliers", {})
    assert isinstance(out, dict)

    soft = out.get("llm.int8() outliers soft difinition", [])
    hard = out.get("llm.int8() outliers hard definition", [])
    iqr = out.get("interquantile_outliers", [])
    token_count = int(out.get("token_count", 0))

    assert len(soft) == token_count
    assert len(hard) == token_count
    assert len(iqr) == token_count
    assert "massive_activations" not in out

    detected = out.get("outliers detected", [])
    assert isinstance(detected, list)
    assert len(detected) == token_count
    assert len(out.get("llm.int8() outliers soft difinition_layer", [])) == token_count

    table = str(out.get("table", ""))
    assert "idx" in table and "token" in table and "outliers detected" in table
    assert _SOFT_FLAG not in table
    assert "interquantile_outliers" not in table
    assert "per_token_max_kurtosis" not in table
    assert "per_token_max_median_ratio" not in table
    assert "mass" not in table

    assert "per_token_max_kurtosis" not in out
    assert "per_token_max_median_ratio" not in out

    report = stats.get("report", {})
    assert isinstance(report, dict)
    assert "max_median_ratio" in report
    assert "kurtosis" in report
    abs_tops = report.get("activation_abs", {})
    assert isinstance(abs_tops, dict)
    for key in (
        "top_1",
        "top_2",
        "top_3",
        "top_10",
        "top_p01",
        "top_p10",
        "top_p50",
        "top_p90",
        "top_p99",
    ):
        assert key in abs_tops

    first_layer = next(iter(stats["layers"].values()))
    kurt = first_layer.get("kurtosis", [])
    assert isinstance(kurt, list) and kurt
    assert any(v is not None for v in kurt)
