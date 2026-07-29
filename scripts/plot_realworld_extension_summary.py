from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISON_CSV = Path(
    "runs/realworld_extension_comparison_summary.csv"
)
PAIRNORM_CSV = Path(
    "runs/realworld_extension_pairnorm_effects.csv"
)
PLOT_DIR = Path(
    "plots/realworld_extension_summary"
)


def plot_depth_drop(comparison):
    frame = comparison.sort_values(
        "baseline_depth_drop"
    ).reset_index(drop=True)

    positions = np.arange(len(frame))
    width = 0.38

    plt.figure(figsize=(12, 7))

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
        label="Full",
    )

    plt.yticks(
        positions,
        frame["dataset"],
    )
    plt.xlabel("Accuracy decrease (percentage points)")
    plt.title(
        "Depth-related performance decrease across datasets"
    )
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "depth_drop_across_datasets.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_pairnorm_gain(pairnorm):
    pivot = pairnorm.pivot(
        index="dataset",
        columns="phase",
        values="pairnorm_gain_percentage_points",
    )

    pivot = pivot.sort_values("Phase 2")

    positions = np.arange(len(pivot))
    width = 0.38

    plt.figure(figsize=(12, 7))

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

    plt.axvline(0.0, linewidth=1)
    plt.yticks(
        positions,
        pivot.index,
    )
    plt.xlabel(
        "Best PairNorm minus best baseline "
        "(percentage points)"
    )
    plt.title(
        "PairNorm effect across datasets"
    )
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "pairnorm_gain_across_datasets.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = pd.read_csv(COMPARISON_CSV)
    pairnorm = pd.read_csv(PAIRNORM_CSV)

    plot_depth_drop(comparison)
    plot_pairnorm_gain(pairnorm)


if __name__ == "__main__":
    main()
