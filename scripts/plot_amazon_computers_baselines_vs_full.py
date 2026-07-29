import os
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = os.path.expanduser("plots/amazon_computers_baselines_vs_full")
os.makedirs(plot_dir, exist_ok=True)

for phase_name, base_csv, full_csv in [
    ("Phase1",
     "runs/amazon_computers_phase1_baselines/amazon_computers_phase1_baselines_grouped.csv",
     "runs/amazon_computers_phase1_full/amazon_computers_phase1_full_grouped.csv"),
    ("Phase2",
     "runs/amazon_computers_phase2_baselines/amazon_computers_phase2_baselines_grouped.csv",
     "runs/amazon_computers_phase2_full/amazon_computers_phase2_full_grouped.csv"),
]:
    base_df = pd.read_csv(base_csv)
    full_df = pd.read_csv(full_csv)

    best_base = base_df.sort_values(
        by=["mean_best_test_acc", "mean_best_val_acc"],
        ascending=[False, False]
    ).iloc[0]

    best_full = full_df.sort_values(
        by=["mean_best_test_acc", "mean_best_val_acc"],
        ascending=[False, False]
    ).iloc[0]

    plot_df = pd.DataFrame([
        {
            "label": f"Baselines\n{best_base['model']}\nL={int(best_base['num_layers'])}, H={int(best_base['hidden_channels'])}",
            "score": best_base["mean_best_test_acc"],
            "std": best_base["std_best_test_acc"],
        },
        {
            "label": f"Full\n{best_full['model']}\nL={int(best_full['num_layers'])}, H={int(best_full['hidden_channels'])}",
            "score": best_full["mean_best_test_acc"],
            "std": best_full["std_best_test_acc"],
        },
    ])

    plt.figure(figsize=(9, 6))
    plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Amazon Computers - {phase_name}: Baselines vs Full")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"amazon_computers_{phase_name}_baselines_vs_full.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
