import os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = os.path.expanduser("runs/actor_phase1_baselines")
plot_dir = os.path.expanduser("plots/actor_phase1_baselines")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(os.path.join(run_dir, "actor_phase1_baselines_grouped.csv"))

plt.figure(figsize=(11, 7))
for model_name in sorted(df["model"].unique()):
    for hidden_channels in sorted(df["hidden_channels"].unique()):
        sub = df[(df["model"] == model_name) & (df["hidden_channels"] == hidden_channels)].sort_values("num_layers")
        if len(sub) == 0:
            continue
        plt.errorbar(
            sub["num_layers"],
            sub["mean_best_test_acc"],
            yerr=sub["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=f"{model_name} (H={hidden_channels})",
        )

plt.xlabel("Number of Layers")
plt.ylabel("Mean Best Test Accuracy")
plt.title("Actor Phase 1 Baselines")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "actor_phase1_baselines_depth.png"))
plt.close()

plt.figure(figsize=(11, 6))
rows = []
for model_name in sorted(df["model"].unique()):
    best = df[df["model"] == model_name].sort_values(
        by=["mean_best_test_acc", "mean_best_val_acc"],
        ascending=[False, False]
    ).iloc[0]
    rows.append({
        "label": f"{model_name}\nL={int(best['num_layers'])}, H={int(best['hidden_channels'])}",
        "score": best["mean_best_test_acc"],
        "std": best["std_best_test_acc"],
    })

plot_df = pd.DataFrame(rows)
plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
plt.ylabel("Mean Best Test Accuracy")
plt.title("Actor Phase 1 Baselines - Best Config per Model")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "actor_phase1_baselines_best_bar.png"))
plt.close()

print(f"Saved plots to: {plot_dir}")
