import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_ring_vs_crossed_ring_grouped.csv")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/synthetic_ring_vs_crossed_ring")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Plot 1: pro Modell, alle Linien (Topologie + Hidden)
for model_name in sorted(df["model"].unique()):
    plt.figure(figsize=(11, 7))
    subset = df[df["model"] == model_name]

    for topology in sorted(subset["topology"].unique()):
        topo_subset = subset[subset["topology"] == topology]

        for hidden_channels in sorted(topo_subset["hidden_channels"].unique()):
            line_subset = topo_subset[topo_subset["hidden_channels"] == hidden_channels].sort_values("distance")

            plt.errorbar(
                line_subset["distance"],
                line_subset["mean_best_test_acc"],
                yerr=line_subset["std_best_test_acc"],
                marker="o",
                capsize=4,
                label=f"{topology} (H={hidden_channels})",
            )

    chance_level = 1.0 / 8.0
    plt.axhline(y=chance_level, linestyle="--", label="Chance (0.125)")
    plt.xlabel("Distance")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Ring vs CrossedRing - {model_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Ring_vs_CrossedRing_{model_name}_all_lines.png"))
    plt.close()

# Plot 2: pro Modell, beste hidden size pro Distanz und Topologie
best_hidden_df = (
    df.sort_values(
        by=["model", "topology", "distance", "mean_best_test_acc", "mean_final_test_acc"],
        ascending=[True, True, True, False, False]
    )
    .groupby(["model", "topology", "distance"], as_index=False)
    .first()
)

for model_name in sorted(best_hidden_df["model"].unique()):
    plt.figure(figsize=(10, 6))
    subset = best_hidden_df[best_hidden_df["model"] == model_name]

    for topology in sorted(subset["topology"].unique()):
        line_subset = subset[subset["topology"] == topology].sort_values("distance")

        plt.errorbar(
            line_subset["distance"],
            line_subset["mean_best_test_acc"],
            yerr=line_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=topology,
        )

        for _, row in line_subset.iterrows():
            plt.annotate(
                f"H={int(row['hidden_channels'])}",
                (row["distance"], row["mean_best_test_acc"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    chance_level = 1.0 / 8.0
    plt.axhline(y=chance_level, linestyle="--", label="Chance (0.125)")
    plt.xlabel("Distance")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Ring vs CrossedRing (best hidden per distance) - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Ring_vs_CrossedRing_{model_name}_best_hidden.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
