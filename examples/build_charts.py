from __future__ import annotations

from pathlib import Path

from acta.visualizer import build_charts_from_stats_file


def main() -> None:
    stats_path = Path("./distilbert_activations_analysis").expanduser().resolve()
    if stats_path.is_dir():
        runs = sorted([p for p in stats_path.iterdir() if p.is_dir()])
        if not runs:
            raise FileNotFoundError(
                f"No run directories found under {stats_path}. Run an eval example first."
            )
        stats_file = runs[-1] / "stats.json"
    else:
        stats_file = stats_path

    output_dir = build_charts_from_stats_file(stats_file)
    print(f"Charts created in: {output_dir}")


if __name__ == "__main__":
    main()
