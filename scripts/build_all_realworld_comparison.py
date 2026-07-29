from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs")

OUTPUT_BEST = RUNS_DIR / "all_realworld_best_configs.csv"
OUTPUT_COMPARISON = RUNS_DIR / "all_realworld_comparison.csv"
OUTPUT_PAIRNORM = RUNS_DIR / "all_realworld_pairnorm_best_grid.csv"
OUTPUT_MATCHED_DETAIL = RUNS_DIR / "all_realworld_pairnorm_matched_detail.csv"
OUTPUT_MATCHED_SUMMARY = RUNS_DIR / "all_realworld_pairnorm_matched_summary.csv"
OUTPUT_INCOMPLETE = RUNS_DIR / "all_realworld_incomplete_datasets.csv"

BLOCKS = [
    "phase1_baselines",
    "phase2_baselines",
    "phase1_full",
    "phase2_full",
]

BLOCK_LABELS = {
    "phase1_baselines": "Phase 1 Baselines",
    "phase2_baselines": "Phase 2 Baselines",
    "phase1_full": "Phase 1 Full",
    "phase2_full": "Phase 2 Full",
}

REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "num_layers",
    "hidden_channels",
    "mean_best_test_acc",
    "std_best_test_acc",
    "mean_best_val_acc",
}


def detect_block(path: Path):
    normalized = str(path).lower().replace("-", "_")

    for block in BLOCKS:
        if block in normalized:
            return block

    return None


def best_row(df: pd.DataFrame) -> pd.Series:
    return (
        df.sort_values(
            [
                "mean_best_test_acc",
                "mean_best_val_acc",
            ],
            ascending=[False, False],
        )
        .iloc[0]
    )


def discover_grouped_files():
    candidates = {}

    for path in RUNS_DIR.rglob("*_grouped.csv"):
        filename = path.name.lower()

        # Best-of-one and already generated global summaries skippen.
        if "best_grouped" in filename:
            continue

        if filename.startswith("all_realworld"):
            continue

        if filename.startswith("realworld_extension"):
            continue

        block = detect_block(path)

        if block is None:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable file {path}: {exc}")
            continue

        missing = REQUIRED_COLUMNS - set(df.columns)

        if missing:
            continue

        for dataset, dataset_df in df.groupby("dataset"):
            dataset_df = dataset_df.copy().reset_index(drop=True)

            key = (str(dataset), block)

            candidate = {
                "df": dataset_df,
                "path": path,
                "num_rows": len(dataset_df),
                "mtime": path.stat().st_mtime,
            }

            previous = candidates.get(key)

            # Bei mehreren möglichen Dateien wird die vollständigere,
            # danach die neuere Datei verwendet.
            if previous is None:
                candidates[key] = candidate
            else:
                new_rank = (
                    candidate["num_rows"],
                    candidate["mtime"],
                )
                old_rank = (
                    previous["num_rows"],
                    previous["mtime"],
                )

                if new_rank > old_rank:
                    candidates[key] = candidate

    return candidates


def build_best_config_table(candidates):
    rows = []

    for (dataset, block), candidate in sorted(candidates.items()):
        row = best_row(candidate["df"]).to_dict()

        row["dataset"] = dataset
        row["block"] = block
        row["block_label"] = BLOCK_LABELS[block]
        row["source_file"] = str(candidate["path"])

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "Keine passenden *_grouped.csv-Dateien gefunden."
        )

    preferred_columns = [
        "dataset",
        "block",
        "block_label",
        "model",
        "num_layers",
        "hidden_channels",
        "mean_best_test_acc",
        "std_best_test_acc",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_final_test_acc",
        "std_final_test_acc",
        "mean_best_epoch",
        "std_best_epoch",
        "num_splits",
        "source_file",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in result.columns
    ]

    return result[available_columns]


def build_comparison(best_df):
    score_table = best_df.pivot(
        index="dataset",
        columns="block",
        values="mean_best_test_acc",
    )

    incomplete_rows = []

    for dataset in score_table.index:
        missing_blocks = [
            block
            for block in BLOCKS
            if (
                block not in score_table.columns
                or pd.isna(score_table.loc[dataset, block])
            )
        ]

        if missing_blocks:
            incomplete_rows.append(
                {
                    "dataset": dataset,
                    "missing_blocks": ";".join(missing_blocks),
                }
            )

    incomplete_df = pd.DataFrame(incomplete_rows)

    complete = score_table.dropna(
        subset=BLOCKS,
    ).copy()

    comparison = pd.DataFrame(
        {
            "dataset": complete.index,
            "phase1_baseline": complete[
                "phase1_baselines"
            ].values,
            "phase2_baseline": complete[
                "phase2_baselines"
            ].values,
            "phase1_full": complete[
                "phase1_full"
            ].values,
            "phase2_full": complete[
                "phase2_full"
            ].values,
        }
    )

    comparison["baseline_depth_drop"] = (
        comparison["phase1_baseline"]
        - comparison["phase2_baseline"]
    )

    comparison["full_depth_drop"] = (
        comparison["phase1_full"]
        - comparison["phase2_full"]
    )

    comparison["depth_drop_reduction"] = (
        comparison["baseline_depth_drop"]
        - comparison["full_depth_drop"]
    )

    comparison["full_gain_phase1"] = (
        comparison["phase1_full"]
        - comparison["phase1_baseline"]
    )

    comparison["full_gain_phase2"] = (
        comparison["phase2_full"]
        - comparison["phase2_baseline"]
    )

    comparison["depth_drop_reduction_fraction"] = np.where(
        comparison["baseline_depth_drop"].abs() > 1e-12,
        comparison["depth_drop_reduction"]
        / comparison["baseline_depth_drop"],
        np.nan,
    )

    comparison = comparison.sort_values(
        "dataset"
    ).reset_index(drop=True)

    return comparison, incomplete_df


def build_pairnorm_tables(candidates):
    best_grid_rows = []
    matched_rows = []

    for dataset in sorted(
        {key[0] for key in candidates}
    ):
        for phase_number in [1, 2]:
            baseline_block = f"phase{phase_number}_baselines"
            full_block = f"phase{phase_number}_full"

            baseline_candidate = candidates.get(
                (dataset, baseline_block)
            )
            full_candidate = candidates.get(
                (dataset, full_block)
            )

            if baseline_candidate is None or full_candidate is None:
                continue

            baseline_df = baseline_candidate["df"]
            full_df = full_candidate["df"]

            pairnorm_df = full_df[
                full_df["model"].astype(str).str.endswith(
                    "PairNorm"
                )
            ].copy()

            if pairnorm_df.empty:
                continue

            best_baseline = best_row(baseline_df)
            best_pairnorm = best_row(pairnorm_df)
            best_overall = best_row(full_df)

            gain = (
                best_pairnorm["mean_best_test_acc"]
                - best_baseline["mean_best_test_acc"]
            )

            best_grid_rows.append(
                {
                    "dataset": dataset,
                    "phase": f"Phase {phase_number}",
                    "baseline_model": best_baseline["model"],
                    "baseline_layers": int(
                        best_baseline["num_layers"]
                    ),
                    "baseline_hidden": int(
                        best_baseline["hidden_channels"]
                    ),
                    "baseline_mean_test": best_baseline[
                        "mean_best_test_acc"
                    ],
                    "baseline_std_test": best_baseline[
                        "std_best_test_acc"
                    ],
                    "pairnorm_model": best_pairnorm["model"],
                    "pairnorm_layers": int(
                        best_pairnorm["num_layers"]
                    ),
                    "pairnorm_hidden": int(
                        best_pairnorm["hidden_channels"]
                    ),
                    "pairnorm_mean_test": best_pairnorm[
                        "mean_best_test_acc"
                    ],
                    "pairnorm_std_test": best_pairnorm[
                        "std_best_test_acc"
                    ],
                    "pairnorm_gain": gain,
                    "pairnorm_gain_percentage_points": 100.0 * gain,
                    "overall_winner": best_overall["model"],
                    "pairnorm_is_overall_winner": str(
                        best_overall["model"]
                    ).endswith("PairNorm"),
                }
            )

            # Kontrollierter Vergleich:
            # PairNorm gegen dasselbe Basismodell bei gleicher
            # Layerzahl und gleicher Hidden-Dimension.
            for _, pairnorm_row in pairnorm_df.iterrows():
                pairnorm_model = str(pairnorm_row["model"])
                baseline_model = pairnorm_model.removesuffix(
                    "PairNorm"
                )

                matched_baseline = full_df[
                    (full_df["model"] == baseline_model)
                    & (
                        full_df["num_layers"]
                        == pairnorm_row["num_layers"]
                    )
                    & (
                        full_df["hidden_channels"]
                        == pairnorm_row["hidden_channels"]
                    )
                ]

                if matched_baseline.empty:
                    continue

                baseline_row = matched_baseline.iloc[0]

                matched_gain = (
                    pairnorm_row["mean_best_test_acc"]
                    - baseline_row["mean_best_test_acc"]
                )

                matched_rows.append(
                    {
                        "dataset": dataset,
                        "phase": f"Phase {phase_number}",
                        "baseline_model": baseline_model,
                        "pairnorm_model": pairnorm_model,
                        "num_layers": int(
                            pairnorm_row["num_layers"]
                        ),
                        "hidden_channels": int(
                            pairnorm_row["hidden_channels"]
                        ),
                        "baseline_mean_test": baseline_row[
                            "mean_best_test_acc"
                        ],
                        "pairnorm_mean_test": pairnorm_row[
                            "mean_best_test_acc"
                        ],
                        "matched_gain": matched_gain,
                        "matched_gain_percentage_points": (
                            100.0 * matched_gain
                        ),
                    }
                )

    best_grid_df = pd.DataFrame(best_grid_rows)
    matched_detail_df = pd.DataFrame(matched_rows)

    if matched_detail_df.empty:
        matched_summary_df = pd.DataFrame()
    else:
        matched_summary_df = (
            matched_detail_df.groupby(
                ["dataset", "phase"],
                as_index=False,
            )
            .agg(
                mean_matched_gain=(
                    "matched_gain",
                    "mean",
                ),
                median_matched_gain=(
                    "matched_gain",
                    "median",
                ),
                best_matched_gain=(
                    "matched_gain",
                    "max",
                ),
                worst_matched_gain=(
                    "matched_gain",
                    "min",
                ),
                positive_configurations=(
                    "matched_gain",
                    lambda values: int((values > 0).sum()),
                ),
                num_configurations=(
                    "matched_gain",
                    "count",
                ),
            )
        )

        for column in [
            "mean_matched_gain",
            "median_matched_gain",
            "best_matched_gain",
            "worst_matched_gain",
        ]:
            matched_summary_df[
                f"{column}_percentage_points"
            ] = 100.0 * matched_summary_df[column]

    return (
        best_grid_df,
        matched_detail_df,
        matched_summary_df,
    )


def print_summary(comparison, pairnorm_df, incomplete_df):
    print("\n=== Complete real-world datasets ===")
    print(", ".join(comparison["dataset"].tolist()))

    print(
        f"\nNumber of complete datasets: {len(comparison)}"
    )

    print("\n=== Mean cross-dataset effects ===")

    print(
        "Mean baseline depth drop: "
        f"{100.0 * comparison['baseline_depth_drop'].mean():.2f} pp"
    )

    print(
        "Median baseline depth drop: "
        f"{100.0 * comparison['baseline_depth_drop'].median():.2f} pp"
    )

    print(
        "Mean full depth drop: "
        f"{100.0 * comparison['full_depth_drop'].mean():.2f} pp"
    )

    if not pairnorm_df.empty:
        for phase in ["Phase 1", "Phase 2"]:
            phase_df = pairnorm_df[
                pairnorm_df["phase"] == phase
            ]

            if phase_df.empty:
                continue

            wins = int(
                phase_df["pairnorm_is_overall_winner"].sum()
            )

            print(
                f"{phase} PairNorm overall wins: "
                f"{wins}/{len(phase_df)}"
            )

            print(
                f"{phase} mean PairNorm gain: "
                f"{phase_df['pairnorm_gain_percentage_points'].mean():.2f} pp"
            )

            print(
                f"{phase} median PairNorm gain: "
                f"{phase_df['pairnorm_gain_percentage_points'].median():.2f} pp"
            )

    if not incomplete_df.empty:
        print("\n=== Datasets with missing blocks ===")
        print(incomplete_df.to_string(index=False))


def main():
    candidates = discover_grouped_files()

    best_df = build_best_config_table(candidates)
    best_df.to_csv(OUTPUT_BEST, index=False)

    comparison, incomplete_df = build_comparison(best_df)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    incomplete_df.to_csv(OUTPUT_INCOMPLETE, index=False)

    (
        pairnorm_df,
        matched_detail_df,
        matched_summary_df,
    ) = build_pairnorm_tables(candidates)

    pairnorm_df.to_csv(OUTPUT_PAIRNORM, index=False)
    matched_detail_df.to_csv(
        OUTPUT_MATCHED_DETAIL,
        index=False,
    )
    matched_summary_df.to_csv(
        OUTPUT_MATCHED_SUMMARY,
        index=False,
    )

    print(f"Saved: {OUTPUT_BEST}")
    print(f"Saved: {OUTPUT_COMPARISON}")
    print(f"Saved: {OUTPUT_PAIRNORM}")
    print(f"Saved: {OUTPUT_MATCHED_DETAIL}")
    print(f"Saved: {OUTPUT_MATCHED_SUMMARY}")
    print(f"Saved: {OUTPUT_INCOMPLETE}")

    print_summary(
        comparison,
        pairnorm_df,
        incomplete_df,
    )


if __name__ == "__main__":
    main()
