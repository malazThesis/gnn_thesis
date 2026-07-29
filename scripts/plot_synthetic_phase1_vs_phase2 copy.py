import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_phase1_vs_phase2_grouped.csv")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/synthetic_phase1_vs_phase2")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(csv_path)

topologies = sorted(df["topology"].unique())
models = sorted(df["model"].unique())

for topology in topologies:
    for model in models:
        subset = df[(df["topology"] == topology) & (df["model"] == model)].copy()

        phase1 = subset[subset["phase"] == "Phase1"].copy()
        phase2 = subset[subset["phase"] == "Phase2"].copy()

        # Phase 1: beste hidden dimension pro Distanz
        phase1_best = (
            phase1.sort_values(
                by=["distance", "mean_best_test_acc", "mean_final_test_acc"],
                ascending=[True, False, False]
            )
            .groupby(["distance"], as_index=False)
            .first()
            .sort_values("distance")
        )

        # Phase 2: beste Tiefe pro Distanz
        phase2_best = (
            phase2.sort_values(
                by=["distance", "mean_best_test_acc", "mean_final_test_acc"],
                ascending=[True, False, False]
            )
            .groupby(["distance"], as_index=False)
            .first()
            .sort_values("distance")
        )

        # Gemeinsame Distanzmenge
        distances = sorted(set(phase1_best["distance"]).intersection(set(phase2_best["distance"])))
        if len(distances) == 0:
            continue

        p1 = phase1_best[phase1_best["distance"].isin(distances)].sort_values("distance")
        p2 = phase2_best[phase2_best["distance"].isin(distances)].sort_values("distance")

        # Plot 1: Linienvergleich
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            p1["distance"],
            p1["mean_best_test_acc"],
            yerr=p1["std_best_test_acc"],
            marker="o",
            capsize=4,
            label="Phase 1 (best hidden)",
        )
        plt.errorbar(
            p2["distance"],
            p2["mean_best_test_acc"],
            yerr=p2["std_best_test_acc"],
            marker="o",
            capsize=4,
            label="Phase 2 (best depth)",
        )

        for _, row in p1.iterrows():
            plt.annotate(
                f"H={int(row['hidden_channels'])}",
                (row["distance"], row["mean_best_test_acc"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        for _, row in p2.iterrows():
            plt.annotate(
                f"L={int(row['num_layers'])}",
                (row["distance"], row["mean_best_test_acc"]),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=8,
            )

        plt.axhline(y=1.0/8.0, linestyle="--", label="Chance (0.125)")
        plt.xlabel("Distance")
        plt.ylabel("Mean Best Test Accuracy")
        plt.title(f"{topology} - {model}: Phase 1 vs Phase 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{topology}_{model}_phase1_vs_phase2_line.png"))
        plt.close()

        # Plot 2: Balkenvergleich
        plt.figure(figsize=(11, 6))
        x = list(range(len(distances)))
        width = 0.35

        plt.bar(
            [i - width/2 for i in x],
            p1["mean_best_test_acc"],
            width=width,
            yerr=p1["std_best_test_acc"],
            capsize=4,
            label="Phase 1",
        )
        plt.bar(
            [i + width/2 for i in x],
            p2["mean_best_test_acc"],
            width=width,
            yerr=p2["std_best_test_acc"],
            capsize=4,
            label="Phase 2",
        )

        labels = []
        for (_, r1), (_, r2) in zip(p1.iterrows(), p2.iterrows()):
            labels.append(
                f"D={int(r1['distance'])}\nP1:H={int(r1['hidden_channels'])}\nP2:L={int(r2['num_layers'])}"
            )

        plt.xticks(x, labels)
        plt.ylabel("Mean Best Test Accuracy")
        plt.title(f"{topology} - {model}: Best Phase 1 vs Best Phase 2")
        plt.axhline(y=1.0/8.0, linestyle="--", label="Chance (0.125)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{topology}_{model}_phase1_vs_phase2_bar.png"))
        plt.close()

print(f"Saved plots to: {plot_dir}")
