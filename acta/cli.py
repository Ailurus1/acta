from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import click
import torch
from torch import nn

from acta import AutoAnalyzer

DUMP_ROOT_NAME = ".acta_dump_results"


def _package_version() -> str:
    try:
        return importlib.metadata.version("acta")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def dump_results_root() -> Path:
    return (Path.cwd() / DUMP_ROOT_NAME).resolve()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _dict_values_only_tensors(d: dict[Any, Any]) -> bool:
    return bool(d) and all(isinstance(v, torch.Tensor) for v in d.values())


def _is_weights_only_checkpoint(obj: Any) -> bool:
    if isinstance(obj, nn.Module):
        return False
    if not isinstance(obj, dict):
        return False
    if _dict_values_only_tensors(obj):
        return True
    sd = obj.get("state_dict")
    if isinstance(sd, dict) and sd and _dict_values_only_tensors(sd):
        if not any(isinstance(v, nn.Module) for v in obj.values()):
            return True
    return False


def extract_nn_module(obj: Any) -> nn.Module:
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
        if _is_weights_only_checkpoint(obj):
            raise click.ClickException(
                "The checkpoint appears to contain weights only (e.g. a state_dict), "
                "not a full nn.Module topology. Save a serialized model object instead."
            )
        raise click.ClickException(
            "Could not find an nn.Module in this file. Expected a pickled module or a "
            "mapping that contains one (e.g. {'model': <nn.Module>})."
        )
    raise click.ClickException(
        f"Unsupported checkpoint type {type(obj).__name__!r}; expected nn.Module."
    )


def _experiment_run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "stats.json").is_file():
            runs.append(p)
    return runs


def _summarize_run(run_dir: Path) -> str:
    stats_path = run_dir / "stats.json"
    lines = [
        f"Results directory: {run_dir}",
        f"Statistics file:   {stats_path}",
    ]
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        lines.append(f"  (could not read statistics: {e})")
        return "\n".join(lines)

    meta_raw = data.get("_acta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    mn = meta.get("model_name")
    if isinstance(mn, str) and mn.strip():
        lines.append(f"Model name:          {mn.strip()}")
    elif mn is not None:
        lines.append(f"Model name:          {mn}")
    else:
        lines.append("Model name:          (not recorded)")

    layers = data.get("layers")
    n_layers = len(layers) if isinstance(layers, dict) else 0
    lines.append(f"Tracked layers (stats): {n_layers}")

    if meta:
        ts = meta.get("session_timestamp")
        if ts:
            lines.append(f"Session timestamp: {ts}")

    outliers = data.get("outliers")
    if isinstance(outliers, dict):
        tc = outliers.get("token_count")
        if tc is not None:
            lines.append(f"Outlier payload token_count: {tc}")

    return "\n".join(lines)


COMMAND_HELP = """
Commands:

  show   List Acta experiment runs under ./.acta_dump_results.

  apply  Load an nn.Module (or checkpoint containing one) from a .pt file,
         wrap it with Acta hooks via AutoAnalyzer, and torch.save the result.
         Refuses plain state_dict checkpoints. If --path-to-save is omitted,
         the output path equals the input file and you must confirm overwrite.

  clear  Delete the entire ./.acta_dump_results directory after confirmation.

  plot   Build charts from a stats.json path using the acta.visualizer.
         Charts default to the same directory as the stats file unless
         --path-to-charts is set.

  help   Print this overview (same as `acta help`).
"""


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_package_version(), prog_name="acta")
def main() -> None:
    pass

@main.command("show")
def show_cmd() -> None:
    root = dump_results_root()
    runs = _experiment_run_dirs(root)
    if not runs:
        click.echo(
            f"No experiment runs found under {root}\n"
            "(each run is a subdirectory containing stats.json)."
        )
        return
    click.echo(f"Experiments root: {root}\n")
    for i, run_dir in enumerate(runs):
        if i:
            click.echo()
        click.echo(_summarize_run(run_dir))


@main.command("apply")
@click.argument(
    "path_to_model_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--path-to-save",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output .pt path (defaults to the input path).",
)
@click.option(
    "--target-layers",
    default=None,
    show_default=False,
    help='Hook pattern(s), same as acta.AutoAnalyzer target_layers (default: all leaf modules).',
)
def apply_cmd(
    path_to_model_file: Path,
    path_to_save: Path | None,
    target_layers: str | None,
) -> None:
    src = path_to_model_file.expanduser().resolve()
    dst = (
        path_to_save.expanduser().resolve()
        if path_to_save is not None
        else src
    )

    if dst == src:
        click.confirm(
            f"This will overwrite the input file:\n  {src}\nProceed?",
            abort=True,
            default=False,
        )

    payload = _torch_load(src)
    inner = extract_nn_module(payload)

    tl: str | list[str] | None = (
        None if target_layers is None else target_layers
    )

    tmp = Path(tempfile.mkdtemp(prefix="acta-apply-"))
    try:
        wrapped = AutoAnalyzer(
            inner,
            dump_stats_path=str(tmp / "stats.json"),
            target_layers=tl,
            draw_charts=False,
            verbose=False,
            finalize_on_exit=False,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(wrapped, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    click.echo(f"Saved wrapped model with `acta` hooks to:\n  {dst}")


@main.command("clear")
def clear_cmd() -> None:
    root = dump_results_root()
    if not root.exists():
        click.echo(f"Nothing to clear: {root} does not exist.")
        return
    click.confirm(
        f"Delete all Acta dump results?\n  {root}\nThis cannot be undone.",
        abort=True,
        default=False,
    )
    shutil.rmtree(root)
    click.echo(f"Removed {root}")


@main.command("plot")
@click.argument(
    "path_to_stats",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--path-to-charts",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for charts (default: directory of stats.json).",
)
def plot_cmd(path_to_stats: Path, path_to_charts: Path | None) -> None:
    from acta.visualizer import build_charts_from_stats_file

    stats_path = path_to_stats.expanduser().resolve()
    out = (
        path_to_charts.expanduser().resolve()
        if path_to_charts is not None
        else None
    )
    output_dir = build_charts_from_stats_file(stats_path, output_dir=out)
    click.echo(f"Charts written under:\n  {output_dir}")


@main.command("help")
@click.pass_context
def help_cmd(ctx: click.Context) -> None:
    root_ctx = ctx.parent
    root_name = root_ctx.command_path if root_ctx is not None else "acta"
    click.echo(f"{root_name} — Acta CLI\n")
    if root_ctx is not None:
        click.echo(root_ctx.command.get_short_help_str())
    click.echo(COMMAND_HELP)


if __name__ == "__main__":
    main()
