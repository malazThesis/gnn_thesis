from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS_DIR = Path("runs")
PLOT_DIR = Path("plots/all_realworld_comparison")

COMPARISON_CSV = RUNS_DIR / "all_realworld_comparison.csv"
PAIRNORM_CSV = RUNS_DIR / "all_realworld_pairnorm_best_grid.csv"
MATCHED_CSV = RUNS_DIR / "all_realworld_pairnorm_matched_summary.csv"


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist: {path}"
        )


def save_current_figure(filename: str):
    output_path = PLOT_DIR / filename
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path}")


def plot_best_scores(comparison: pd.DataFrame):
    frame = comparison.sort_values(
        "phase1_baseline",
        ascending=True,
    ).reset_index(drop=True)

    positions = np.arange(len(frame))
    width = 0.19

    plt.figure(figsize=(14, 10))

    series = [
        ("phase1_baseline", "Phase 1 Baselines"),
        ("phase2_baseline", "Phase 2 Baselines"),
        ("phase1_full", "Phase 1 Full"),
        ("phase2_full", "Phase 2 Full"),
    ]

    offsets = [-1.5, -0.5, 0.5, 1.5]

    for offset, (column, label) in zip(offsets, series):
        plt.barh(
            positions + offset * width,
            frame[column],
            height=width,
            label=label,
        )

    plt.yticks(
        positions,
        frame["dataset"],
    )
    plt.xlabel("Mean best test accuracy")
    plt.title(
        "Best performance across all real-world datasets"
    )
    plt.legend()

    save_current_figure(
        "all_realworld_best_scores.png"
    )


def plot_depth_drop(comparison: pd.DataFrame):
    frame = comparison.sort_values(
        "baseline_depth_drop",
        ascending=True,
    ).reset_index(drop=True)

    positions = np.arange(len(frame))
    width = 0.38

    plt.figure(figsize=(14, 10))

    plt.barh(
        positions - width / 2,
        100.0 * frame["baseline_depth_drop"],
        height=width,
        label="Baselines",
    )

    plt.barh(
        positions + width / 2,
        100.0 * frame["full_depth_drop"],
        height=width,
        label="Full grid",
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )
    plt.yticks(
        positions,
        frame["dataset"],
    )
    plt.xlabel(
        "Phase 1 minus Phase 2 accuracy "
        "(percentage points)"
    )
    plt.title(
        "Depth-related performance decrease "
        "across all real-world datasets"
    )
    plt.legend()

    save_current_figure(
        "all_realworld_depth_drop.png"
    )


def plot_pairnorm_best_grid(pairnorm: pd.DataFrame):
    pivot = pairnorm.pivot(
        index="dataset",
        columns="phase",
        values="pairnorm_gain_percentage_points",
    )

    required_phases = ["Phase 1", "Phase 2"]

    missing = [
        phase
        for phase in required_phases
        if phase not in pivot.columns
    ]

    if missing:
        raise ValueError(
            f"Missing PairNorm phases: {missing}"
        )

    pivot = pivot.sort_values(
        "Phase 2",
        ascending=True,
    )

    positions = np.arange(len(pivot))
    width = 0.38

    plt.figure(figsize=(14, 10))

    plt.barh(
        positions - width / 2,
        pivot["Phase 1"],
        height=width,
        label="Phase 1",
    )

    plt.barh(
        positions + width / 2,
        pivot["Phase 2"],
        height=width,
        label="Phase 2",
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )
    plt.yticks(
        positions,
        pivot.index,
    )
    plt.xlabel(
        "Best PairNorm minus best baseline "
        "(percentage points)"
    )
    plt.title(
        "Best-of-grid PairNorm effect "
        "across all real-world datasets"
    )
    plt.legend()

    save_current_figure(
        "all_realworld_pairnorm_best_grid.png"
    )


def plot_pairnorm_matched(matched: pd.DataFrame):
    if matched.empty:
        print(
            "Matched PairNorm summary is empty; "
            "matched plot skipped."
        )
        return

    value_column = (
        "mean_matched_gain_percentage_points"
    )

    if value_column not in matched.columns:
        raise ValueError(
            f"Missing column in {MATCHED_CSV}: "
            f"{value_column}"
        )

    pivot = matched.pivot(
        index="dataset",
        columns="phase",
        values=value_column,
    )

    required_phases = ["Phase 1", "Phase 2"]

    for phase in required_phases:
        if phase not in pivot.columns:
            pivot[phase] = np.nan

    pivot = pivot.sort_values(
        "Phase 2",
        ascending=True,
        na_position="first",
    )

    positions = np.arange(len(pivot))
    width = 0.38

    plt.figure(figsize=(14, 10))

    plt.barh(
        positions - width / 2,
        pivot["Phase 1"],
        height=width,
        label="Phase 1",
    )

    plt.barh(
        positions + width / 2,
        pivot["Phase 2"],
        height=width,
        label="Phase 2",
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )
    plt.yticks(
        positions,
        pivot.index,
    )
    plt.xlabel(
        "Mean matched PairNorm gain "
        "(percentage points)"
    )
    plt.title(
        "PairNorm effect with matched model, "
        "depth, and hidden size"
    )
    plt.legend()

    save_current_figure(
        "all_realworld_pairnorm_matched.png"
    )


def main():
    require_file(COMPARISON_CSV)
    require_file(PAIRNORM_CSV)
    require_file(MATCHED_CSV)

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = pd.read_csv(COMPARISON_CSV)
    pairnorm = pd.read_csv(PAIRNORM_CSV)
    matched = pd.read_csv(MATCHED_CSV)

    print(
        f"Comparison datasets: "
        f"{comparison['dataset'].nunique()}"
    )
    print(
        f"PairNorm datasets: "
        f"{pairnorm['dataset'].nunique()}"
    )

    plot_best_scores(comparison)
    plot_depth_drop(comparison)
    plot_pairnorm_best_grid(pairnorm)
    plot_pairnorm_matched(matched)


if __name__ == "__main__":
    main()
