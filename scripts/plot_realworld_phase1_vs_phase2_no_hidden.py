import os
import pandas as pd
import matplotlib.pyplot as plt

phase1_path = os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase1_gs/realworld_phase1_gs_grouped.csv")
phase2_path = os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase2/realworld_phase2_grouped.csv")

plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/realworld_phase1_vs_phase2_no_hidden")
os.makedirs(plot_dir, exist_ok=True)

phase1_df = pd.read_csv(phase1_path)
phase2_df = pd.read_csv(phase2_path)

phase1_df["phase"] = "Phase1"
phase2_df["phase"] = "Phase2"

df = pd.concat([phase1_df, phase2_df], ignore_index=True)

# Für jede Kombination aus Datensatz, Modell, Layer die beste Hidden-Dimension wählen
best_hidden_df = (
    df.sort_values(
        by=["dataset", "phase", "model", "num_layers", "mean_best_test_acc", "mean_best_val_acc"],
        ascending=[True, True, True, True, False, False]
    )
    .groupby(["dataset", "phase", "model", "num_layers"], as_index=False)
    .first()
)

# Plot 1: gemeinsame Tiefenkurve über Phase 1 + Phase 2
for dataset_name in sorted(best_hidden_df["dataset"].unique()):
    plt.figure(figsize=(11, 7))
    subset = best_hidden_df[best_hidden_df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        model_subset = subset[subset["model"] == model_name].sort_values("num_layers")

        plt.errorbar(
            model_subset["num_layers"],
            model_subset["mean_best_test_acc"],
            yerr=model_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=model_name,
        )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Phase 1 + Phase 2 Depth Comparison (best hidden size) - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase1_vs_phase2_depth_comparison_no_hidden.png"))
    plt.close()

# Plot 2: gleiche Kurve, aber beste hidden size als Text neben dem Punkt
for dataset_name in sorted(best_hidden_df["dataset"].unique()):
    plt.figure(figsize=(12, 7))
    subset = best_hidden_df[best_hidden_df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        model_subset = subset[subset["model"] == model_name].sort_values("num_layers")

        plt.errorbar(
            model_subset["num_layers"],
            model_subset["mean_best_test_acc"],
            yerr=model_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=model_name,
        )

        for _, row in model_subset.iterrows():
            plt.annotate(
                f"H={int(row['hidden_channels'])}",
                (row["num_layers"], row["mean_best_test_acc"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Phase 1 + Phase 2 Depth Comparison with hidden size labels - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase1_vs_phase2_depth_comparison_hidden_labels.png"))
    plt.close()

# Plot 3: beste Konfiguration pro Modell über beide Phasen
for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(12, 7))
    subset = df[df["dataset"] == dataset_name]

    best_rows = []
    for model_name in sorted(subset["model"].unique()):
        best_row = subset[subset["model"] == model_name].sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]

        best_rows.append({
            "label": f"{model_name}\nL={int(best_row['num_layers'])}, H={int(best_row['hidden_channels'])}",
            "score": best_row["mean_best_test_acc"],
            "std": best_row["std_best_test_acc"],
            "phase": best_row["phase"],
        })

    plot_df = pd.DataFrame(best_rows)

    plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Best Model Configurations across Phase 1 + Phase 2 - {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase1_vs_phase2_best_models_bar.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
