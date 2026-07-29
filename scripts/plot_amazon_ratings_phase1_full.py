import os

import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = "runs/amazon_ratings_phase1_full"
PLOT_DIR = "plots/amazon_ratings_phase1_full"

CSV_PATH = os.path.join(
    RUN_DIR,
    "amazon_ratings_phase1_full_grouped.csv",
)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Alle Modelle nach Tiefe
    plt.figure(figsize=(13, 7))

    for model_name in sorted(df["model"].unique()):
        for hidden_channels in sorted(
            df.loc[
                df["model"] == model_name,
                "hidden_channels",
            ].unique()
        ):
            subset = df[
                (df["model"] == model_name)
                & (df["hidden_channels"] == hidden_channels)
            ].sort_values("num_layers")

            plt.errorbar(
                subset["num_layers"],
                subset["mean_best_test_acc"],
                yerr=subset["std_best_test_acc"],
                marker="o",
                capsize=4,
                label=f"{model_name} (H={hidden_channels})",
            )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title("Amazon-Ratings Phase 1 Full")
    plt.xticks(sorted(df["num_layers"].unique()))
    plt.legend(fontsize=8)
    plt.tight_layout()

    depth_path = os.path.join(
        PLOT_DIR,
        "amazon_ratings_phase1_full_depth.png",
    )

    plt.savefig(depth_path, dpi=300)
    plt.close()

    # Beste Konfiguration pro Modell
    best_rows = []

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

        best_rows.append(
            {
                "label": (
                    f"{model_name}\n"
                    f"L={int(best['num_layers'])}, "
                    f"H={int(best['hidden_channels'])}"
                ),
                "score": best["mean_best_test_acc"],
                "std": best["std_best_test_acc"],
            }
        )

    plot_df = pd.DataFrame(best_rows)

    plt.figure(figsize=(12, 6))

    plt.bar(
        plot_df["label"],
        plot_df["score"],
        yerr=plot_df["std"],
        capsize=4,
    )

    plt.ylabel("Mean Best Test Accuracy")
    plt.title(
        "Amazon-Ratings Phase 1 Full – Best Configuration per Model"
    )
    plt.xticks(rotation=15)
    plt.tight_layout()

    bar_path = os.path.join(
        PLOT_DIR,
        "amazon_ratings_phase1_full_best_bar.png",
    )

    plt.savefig(bar_path, dpi=300)
    plt.close()

    # Nur Baseline-Modelle
    baseline_models = [
        "GCN",
        "GAT",
        "GraphSAGE",
    ]

    plt.figure(figsize=(11, 7))

    for model_name in baseline_models:
        for hidden_channels in sorted(
            df["hidden_channels"].unique()
        ):
            subset = df[
                (df["model"] == model_name)
                & (df["hidden_channels"] == hidden_channels)
            ].sort_values("num_layers")

            plt.errorbar(
                subset["num_layers"],
                subset["mean_best_test_acc"],
                yerr=subset["std_best_test_acc"],
                marker="o",
                capsize=4,
                label=f"{model_name} (H={hidden_channels})",
            )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title("Amazon-Ratings Phase 1 Full – Baselines Only")
    plt.xticks(sorted(df["num_layers"].unique()))
    plt.legend(fontsize=8)
    plt.tight_layout()

    baseline_path = os.path.join(
        PLOT_DIR,
        "amazon_ratings_phase1_full_baselines_only.png",
    )

    plt.savefig(baseline_path, dpi=300)
    plt.close()

    print(f"Saved: {depth_path}")
    print(f"Saved: {bar_path}")
    print(f"Saved: {baseline_path}")


if __name__ == "__main__":
    main()
