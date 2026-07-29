import os

import matplotlib.pyplot as plt
import pandas as pd


PLOT_DIR = "plots/amazon_ratings_comparisons"
os.makedirs(PLOT_DIR, exist_ok=True)

P1_BASE = pd.read_csv(
    "runs/amazon_ratings_phase1_baselines/"
    "amazon_ratings_phase1_baselines_grouped.csv"
)
P2_BASE = pd.read_csv(
    "runs/amazon_ratings_phase2_baselines/"
    "amazon_ratings_phase2_baselines_grouped.csv"
)
P1_FULL = pd.read_csv(
    "runs/amazon_ratings_phase1_full/"
    "amazon_ratings_phase1_full_grouped.csv"
)
P2_FULL = pd.read_csv(
    "runs/amazon_ratings_phase2_full/"
    "amazon_ratings_phase2_full_grouped.csv"
)


def best_per_model(df):
    rows = []

    for model_name in sorted(df["model"].unique()):
        best = (
            df[df["model"] == model_name]
            .sort_values(
                [
                    "mean_best_test_acc",
                    "mean_best_val_acc",
                ],
                ascending=[False, False],
            )
            .iloc[0]
        )

        rows.append(best)

    return pd.DataFrame(rows)


def plot_phase_comparison(
    phase1,
    phase2,
    output_name,
    title,
):
    p1 = best_per_model(phase1)
    p2 = best_per_model(phase2)

    rows = []

    for phase_name, frame in [
        ("Phase 1", p1),
        ("Phase 2", p2),
    ]:
        for _, row in frame.iterrows():
            rows.append(
                {
                    "label": (
                        f"{row['model']}\n"
                        f"{phase_name}\n"
                        f"L={int(row['num_layers'])}, "
                        f"H={int(row['hidden_channels'])}"
                    ),
                    "score": row["mean_best_test_acc"],
                    "std": row["std_best_test_acc"],
                }
            )

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(14, 7))
    plt.bar(
        plot_df["label"],
        plot_df["score"],
        yerr=plot_df["std"],
        capsize=4,
    )
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, output_name),
        dpi=300,
    )
    plt.close()


def plot_baselines_vs_full(
    baseline_df,
    full_df,
    phase_name,
):
    best_baseline = (
        baseline_df.sort_values(
            [
                "mean_best_test_acc",
                "mean_best_val_acc",
            ],
            ascending=[False, False],
        )
        .iloc[0]
    )

    best_full = (
        full_df.sort_values(
            [
                "mean_best_test_acc",
                "mean_best_val_acc",
            ],
            ascending=[False, False],
        )
        .iloc[0]
    )

    plot_df = pd.DataFrame(
        [
            {
                "label": (
                    "Baselines\n"
                    f"{best_baseline['model']}\n"
                    f"L={int(best_baseline['num_layers'])}, "
                    f"H={int(best_baseline['hidden_channels'])}"
                ),
                "score": best_baseline["mean_best_test_acc"],
                "std": best_baseline["std_best_test_acc"],
            },
            {
                "label": (
                    "Full\n"
                    f"{best_full['model']}\n"
                    f"L={int(best_full['num_layers'])}, "
                    f"H={int(best_full['hidden_channels'])}"
                ),
                "score": best_full["mean_best_test_acc"],
                "std": best_full["std_best_test_acc"],
            },
        ]
    )

    plt.figure(figsize=(9, 6))
    plt.bar(
        plot_df["label"],
        plot_df["score"],
        yerr=plot_df["std"],
        capsize=4,
    )
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(
        f"Amazon-Ratings {phase_name}: Baselines vs Full"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"amazon_ratings_{phase_name.lower().replace(' ', '')}"
            "_baselines_vs_full.png",
        ),
        dpi=300,
    )
    plt.close()


plot_phase_comparison(
    P1_BASE,
    P2_BASE,
    "amazon_ratings_baselines_phase1_vs_phase2.png",
    "Amazon-Ratings Baselines: Phase 1 vs Phase 2",
)

plot_phase_comparison(
    P1_FULL,
    P2_FULL,
    "amazon_ratings_full_phase1_vs_phase2.png",
    "Amazon-Ratings Full: Phase 1 vs Phase 2",
)

plot_baselines_vs_full(
    P1_BASE,
    P1_FULL,
    "Phase 1",
)

plot_baselines_vs_full(
    P2_BASE,
    P2_FULL,
    "Phase 2",
)

print(f"Saved comparison plots to: {PLOT_DIR}")
