from pathlib import Path

import numpy as np
import pandas as pd


FAIR_ROOT = Path(
    "runs/amazon_ratings_fair_lr_800ep"
)

OS_ROOT = Path(
    "runs/amazon_ratings_matched_lr_oversmoothing_800ep"
)

TABLE_DIR = Path("tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Accuracy bei identischer Lernrate 0.003
# ---------------------------------------------------------

accuracy = pd.read_csv(
    FAIR_ROOT
    / "amazon_ratings_fair_lr_800ep_summary.csv"
)

performance = accuracy[
    np.isclose(
        accuracy["learning_rate"],
        0.003,
    )
    & accuracy["model"].isin(
        [
            "GraphSAGE",
            "GraphSAGEPairNorm",
        ]
    )
    & accuracy["num_layers"].isin(
        [4, 8]
    )
].copy()

performance["Modell"] = performance[
    "model"
].replace(
    {
        "GraphSAGE":
            "GraphSAGE",
        "GraphSAGEPairNorm":
            "GraphSAGE + PairNorm",
    }
)

performance["Tiefe"] = performance[
    "num_layers"
].map(
    lambda value: f"L{int(value)}"
)

performance["Validation (%)"] = (
    100.0
    * performance[
        "mean_best_val_acc"
    ]
)

performance["Test (%)"] = (
    100.0
    * performance[
        "mean_test_at_best_val"
    ]
)

performance["Test-Std. (pp)"] = (
    100.0
    * performance[
        "std_test_at_best_val"
    ]
)

performance["Beste Epoche"] = performance[
    "mean_best_epoch"
]

performance_table = performance[
    [
        "Modell",
        "Tiefe",
        "Validation (%)",
        "Test (%)",
        "Test-Std. (pp)",
        "Beste Epoche",
    ]
].sort_values(
    [
        "Tiefe",
        "Modell",
    ]
)

performance_csv = (
    FAIR_ROOT
    / "amazon_ratings_final_performance_table.csv"
)

performance_tex = (
    TABLE_DIR
    / "amazon_ratings_final_performance.tex"
)

performance_table.to_csv(
    performance_csv,
    index=False,
)

performance_table.to_latex(
    performance_tex,
    index=False,
    float_format="%.2f",
    caption=(
        "Amazon-Ratings-Ergebnisse bei einer gemeinsamen "
        "Lernrate von 0,003 und einem Trainingsbudget "
        "von 800 Epochen."
    ),
    label=(
        "tab:amazon_ratings_matched_lr_performance"
    ),
    escape=False,
)


# ---------------------------------------------------------
# 2. Tiefeneffekt
# ---------------------------------------------------------

depth = pd.read_csv(
    FAIR_ROOT
    / "amazon_ratings_matched_lr_depth_800ep_summary.csv"
)

interaction = depth[
    depth["comparison"]
    == "PairNorm depth interaction"
].iloc[0]

depth_table = pd.DataFrame(
    [
        {
            "Vergleich":
                "GraphSAGE: L8 minus L4",
            "Differenz (pp)":
                depth.loc[
                    depth["comparison"]
                    == "GraphSAGE L8 minus L4",
                    "mean_difference_percentage_points",
                ].iloc[0],
        },
        {
            "Vergleich":
                "PairNorm: L8 minus L4",
            "Differenz (pp)":
                depth.loc[
                    depth["comparison"]
                    == "PairNorm L8 minus L4",
                    "mean_difference_percentage_points",
                ].iloc[0],
        },
        {
            "Vergleich":
                "PairNorm-Tiefeninteraktion",
            "Differenz (pp)":
                interaction[
                    "mean_difference_percentage_points"
                ],
        },
    ]
)

depth_table[
    "95-%-KI unten (pp)"
] = [
    100.0
    * depth.loc[
        depth["comparison"]
        == "GraphSAGE L8 minus L4",
        "ci95_low",
    ].iloc[0],

    100.0
    * depth.loc[
        depth["comparison"]
        == "PairNorm L8 minus L4",
        "ci95_low",
    ].iloc[0],

    100.0 * interaction["ci95_low"],
]

depth_table[
    "95-%-KI oben (pp)"
] = [
    100.0
    * depth.loc[
        depth["comparison"]
        == "GraphSAGE L8 minus L4",
        "ci95_high",
    ].iloc[0],

    100.0
    * depth.loc[
        depth["comparison"]
        == "PairNorm L8 minus L4",
        "ci95_high",
    ].iloc[0],

    100.0 * interaction["ci95_high"],
]

depth_table[
    "Exaktes p (Holm)"
] = [
    depth.loc[
        depth["comparison"]
        == "GraphSAGE L8 minus L4",
        "exact_sign_flip_holm_pvalue",
    ].iloc[0],

    depth.loc[
        depth["comparison"]
        == "PairNorm L8 minus L4",
        "exact_sign_flip_holm_pvalue",
    ].iloc[0],

    interaction[
        "exact_sign_flip_holm_pvalue"
    ],
]

depth_table[
    "Cohen dz"
] = [
    depth.loc[
        depth["comparison"]
        == "GraphSAGE L8 minus L4",
        "cohen_dz",
    ].iloc[0],

    depth.loc[
        depth["comparison"]
        == "PairNorm L8 minus L4",
        "cohen_dz",
    ].iloc[0],

    interaction["cohen_dz"],
]

depth_csv = (
    FAIR_ROOT
    / "amazon_ratings_final_depth_table.csv"
)

depth_tex = (
    TABLE_DIR
    / "amazon_ratings_final_depth_effect.tex"
)

depth_table.to_csv(
    depth_csv,
    index=False,
)

depth_table.to_latex(
    depth_tex,
    index=False,
    float_format="%.3f",
    caption=(
        "Gepaarte Tiefeneffekte bei einer gemeinsamen "
        "Lernrate von 0,003."
    ),
    label=(
        "tab:amazon_ratings_depth_effect"
    ),
    escape=False,
)


# ---------------------------------------------------------
# 3. Mechanismus bei L8
# ---------------------------------------------------------

mechanism = pd.read_csv(
    OS_ROOT
    / "amazon_ratings_oversmoothing_fixed_epoch_effects.csv"
)

metric_labels = {
    "normalized_dirichlet_energy":
        "Normalisierte Dirichlet-Energie",

    "effective_rank_ratio":
        "Effektiver Ranganteil",
}

mechanism_table = mechanism[
    (
        mechanism["num_layers"]
        == 8
    )
    & mechanism["metric"].isin(
        metric_labels
    )
].copy()

mechanism_table["Metrik"] = (
    mechanism_table["metric"]
    .map(metric_labels)
)

mechanism_table["Epoche"] = (
    mechanism_table["epoch"]
    .astype(int)
)

mechanism_table["GraphSAGE"] = (
    mechanism_table["baseline_mean"]
)

mechanism_table["PairNorm"] = (
    mechanism_table["pairnorm_mean"]
)

mechanism_table["Verhältnis"] = (
    mechanism_table[
        "pairnorm_to_baseline_ratio"
    ]
)

mechanism_table[
    "Positive Splits"
] = (
    mechanism_table[
        "positive_pairs"
    ].astype(int)
)

mechanism_table[
    "Exaktes p (Holm)"
] = mechanism_table[
    "exact_sign_flip_holm_pvalue"
]

mechanism_table = mechanism_table[
    [
        "Metrik",
        "Epoche",
        "GraphSAGE",
        "PairNorm",
        "Verhältnis",
        "Positive Splits",
        "Exaktes p (Holm)",
    ]
].sort_values(
    [
        "Metrik",
        "Epoche",
    ]
)

mechanism_csv = (
    OS_ROOT
    / "amazon_ratings_final_mechanism_table.csv"
)

mechanism_tex = (
    TABLE_DIR
    / "amazon_ratings_final_mechanism_l8.tex"
)

mechanism_table.to_csv(
    mechanism_csv,
    index=False,
)

mechanism_table.to_latex(
    mechanism_tex,
    index=False,
    float_format="%.3f",
    caption=(
        "Oversmoothing-Metriken der achtlagigen Modelle "
        "bei festen Trainingsepochen und einer gemeinsamen "
        "Lernrate von 0,003."
    ),
    label=(
        "tab:amazon_ratings_mechanism_l8"
    ),
    escape=False,
)


# ---------------------------------------------------------
# 4. Textzusammenfassung
# ---------------------------------------------------------

summary_path = (
    FAIR_ROOT
    / "amazon_ratings_final_evidence_summary.txt"
)

summary_text = f"""Amazon-Ratings final evidence summary

Matched learning rate:
  learning_rate = 0.003
  epochs = 800
  splits = 10

Depth degradation:
  GraphSAGE depth loss:
    {100.0 * interaction['baseline_mean_depth_loss']:.4f} pp

  PairNorm depth loss:
    {100.0 * interaction['pairnorm_mean_depth_loss']:.4f} pp

  Absolute reduction:
    {100.0 * interaction['absolute_depth_loss_reduction']:.4f} pp

  Relative reduction:
    {100.0 * interaction['relative_depth_loss_reduction']:.2f} %

  Interaction exact Holm p:
    {interaction['exact_sign_flip_holm_pvalue']:.10f}

  Interaction Cohen dz:
    {interaction['cohen_dz']:.4f}

Mechanistic robustness:
  Fixed epochs:
    400, 600, 800

  Metrics:
    normalized Dirichlet energy
    effective rank ratio

  All reported L8 effects:
    PairNorm > GraphSAGE on 10/10 splits
"""

summary_path.write_text(
    summary_text,
    encoding="utf-8",
)

print("\n=== PERFORMANCE ===")
print(
    performance_table.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.3f}",
    )
)

print("\n=== DEPTH EFFECT ===")
print(
    depth_table.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)

print("\n=== L8 MECHANISM ===")
print(
    mechanism_table.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)

print("\nSaved:", performance_csv)
print("Saved:", performance_tex)
print("Saved:", depth_csv)
print("Saved:", depth_tex)
print("Saved:", mechanism_csv)
print("Saved:", mechanism_tex)
print("Saved:", summary_path)
