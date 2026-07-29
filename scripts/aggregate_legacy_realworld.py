import json
import os
from pathlib import Path

import pandas as pd


USER = os.environ["USER"]

OLD_RUNS = Path(
    f"/u/{USER}/gnn_thesis_cluster/runs"
)

OUTPUT_RUNS = Path(
    f"/work/log1/{USER}/gnn_thesis_cluster/runs"
)

DATASETS = {
    "Cora": "cora",
    "CiteSeer": "citeseer",
    "PubMed": "pubmed",
}

BASELINE_MODELS = {
    "GCN",
    "GAT",
    "GraphSAGE",
}

FULL_MODELS = {
    "GCN",
    "GAT",
    "GraphSAGE",
    "GCNPairNorm",
    "GATPairNorm",
    "GraphSAGEPairNorm",
}

PHASE_SOURCES = {
    # realworld_phase1_gs wird bevorzugt.
    # realworld_phase1 dient als Fallback für fehlende Runs.
    "phase1": [
        (OLD_RUNS / "realworld_phase1_gs", 0),
        (OLD_RUNS / "realworld_phase1", 1),
    ],
    "phase2": [
        (OLD_RUNS / "realworld_phase2", 0),
    ],
}

BLOCKS = {
    "phase1_baselines": {
        "phase": "phase1",
        "models": BASELINE_MODELS,
        "expected_configs": 12,
    },
    "phase1_full": {
        "phase": "phase1",
        "models": FULL_MODELS,
        "expected_configs": 24,
    },
    "phase2_baselines": {
        "phase": "phase2",
        "models": BASELINE_MODELS,
        "expected_configs": 18,
    },
    "phase2_full": {
        "phase": "phase2",
        "models": FULL_MODELS,
        "expected_configs": 36,
    },
}

REQUIRED_FIELDS = {
    "dataset",
    "model",
    "num_layers",
    "hidden_channels",
    "best_test_acc_at_best_val",
    "best_val_acc",
    "final_test_acc",
    "best_epoch",
}


def normalize(value):
    return (
        str(value)
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def read_phase(dataset_name, phase):
    records = []
    invalid_files = []

    for source_dir, priority in PHASE_SOURCES[phase]:
        if not source_dir.exists():
            continue

        for path in source_dir.glob("*_summary.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    row = json.load(handle)
            except Exception as exc:
                invalid_files.append(
                    {
                        "path": str(path),
                        "error": str(exc),
                    }
                )
                continue

            if normalize(row.get("dataset")) != normalize(dataset_name):
                continue

            missing = REQUIRED_FIELDS - set(row)

            if missing:
                invalid_files.append(
                    {
                        "path": str(path),
                        "error": (
                            "missing fields: "
                            + ", ".join(sorted(missing))
                        ),
                    }
                )
                continue

            row["_source_path"] = str(path)
            row["_source_priority"] = priority

            # Alte Runs verwenden Seed als Wiederholung.
            # Neuere Runs könnten stattdessen split_idx enthalten.
            row["replicate_id"] = row.get(
                "split_idx",
                row.get("seed"),
            )

            records.append(row)

    if not records:
        return pd.DataFrame(), invalid_files

    frame = pd.DataFrame(records)

    deduplication_columns = [
        "dataset",
        "model",
        "num_layers",
        "hidden_channels",
        "replicate_id",
    ]

    frame = (
        frame.sort_values(
            [
                "_source_priority",
                "_source_path",
            ]
        )
        .drop_duplicates(
            subset=deduplication_columns,
            keep="first",
        )
        .reset_index(drop=True)
    )

    return frame, invalid_files


def aggregate(frame):
    grouped = (
        frame.groupby(
            [
                "dataset",
                "model",
                "num_layers",
                "hidden_channels",
            ],
            as_index=False,
        )
        .agg(
            mean_best_test_acc=(
                "best_test_acc_at_best_val",
                "mean",
            ),
            std_best_test_acc=(
                "best_test_acc_at_best_val",
                "std",
            ),
            mean_best_val_acc=(
                "best_val_acc",
                "mean",
            ),
            std_best_val_acc=(
                "best_val_acc",
                "std",
            ),
            mean_final_test_acc=(
                "final_test_acc",
                "mean",
            ),
            std_final_test_acc=(
                "final_test_acc",
                "std",
            ),
            mean_best_epoch=(
                "best_epoch",
                "mean",
            ),
            std_best_epoch=(
                "best_epoch",
                "std",
            ),
            num_splits=(
                "replicate_id",
                "nunique",
            ),
        )
    )

    return grouped.sort_values(
        [
            "model",
            "num_layers",
            "hidden_channels",
        ]
    ).reset_index(drop=True)


def select_best(grouped):
    return (
        grouped.sort_values(
            [
                "mean_best_test_acc",
                "mean_best_val_acc",
            ],
            ascending=[False, False],
        )
        .head(1)
        .reset_index(drop=True)
    )


def main():
    coverage_rows = []
    invalid_rows = []

    phase_cache = {}

    for dataset_name, dataset_slug in DATASETS.items():
        for phase in ["phase1", "phase2"]:
            frame, invalid = read_phase(
                dataset_name,
                phase,
            )

            phase_cache[(dataset_name, phase)] = frame

            for row in invalid:
                row["dataset"] = dataset_name
                row["phase"] = phase
                invalid_rows.append(row)

            print(
                f"{dataset_name:10s} {phase}: "
                f"{len(frame)} unique runs"
            )

    for dataset_name, dataset_slug in DATASETS.items():
        for block_name, specification in BLOCKS.items():
            phase = specification["phase"]
            allowed_models = specification["models"]

            source = phase_cache[
                (dataset_name, phase)
            ]

            block_frame = source[
                source["model"].isin(allowed_models)
            ].copy()

            if block_frame.empty:
                print(
                    f"ERROR: no rows for "
                    f"{dataset_name} {block_name}"
                )
                continue

            output_dir = (
                OUTPUT_RUNS
                / f"{dataset_slug}_{block_name}"
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            grouped = aggregate(block_frame)
            best = select_best(grouped)

            prefix = f"{dataset_slug}_{block_name}"

            raw_path = (
                output_dir
                / f"{prefix}_summary.csv"
            )
            grouped_path = (
                output_dir
                / f"{prefix}_grouped.csv"
            )
            best_path = (
                output_dir
                / f"{prefix}_best_grouped.csv"
            )

            block_frame.to_csv(
                raw_path,
                index=False,
            )
            grouped.to_csv(
                grouped_path,
                index=False,
            )
            best.to_csv(
                best_path,
                index=False,
            )

            num_configs = len(grouped)
            expected_configs = specification[
                "expected_configs"
            ]

            min_replicates = int(
                grouped["num_splits"].min()
            )
            max_replicates = int(
                grouped["num_splits"].max()
            )

            coverage_rows.append(
                {
                    "dataset": dataset_name,
                    "block": block_name,
                    "num_runs": len(block_frame),
                    "num_configs": num_configs,
                    "expected_configs": expected_configs,
                    "min_replicates": min_replicates,
                    "max_replicates": max_replicates,
                    "complete_config_grid": (
                        num_configs == expected_configs
                    ),
                }
            )

            status = (
                "OK"
                if num_configs == expected_configs
                else "INCOMPLETE"
            )

            print(
                f"{dataset_name:10s} "
                f"{block_name:18s}: "
                f"runs={len(block_frame):3d}, "
                f"configs={num_configs:2d}/"
                f"{expected_configs:2d}, "
                f"replicates={min_replicates}-"
                f"{max_replicates} [{status}]"
            )

            print(
                "  best:",
                best.iloc[0]["model"],
                f"L{int(best.iloc[0]['num_layers'])}",
                f"H{int(best.iloc[0]['hidden_channels'])}",
                f"{best.iloc[0]['mean_best_test_acc']:.4f}",
            )

    coverage = pd.DataFrame(coverage_rows)

    coverage_path = (
        OUTPUT_RUNS
        / "legacy_realworld_coverage.csv"
    )
    coverage.to_csv(
        coverage_path,
        index=False,
    )

    invalid_path = (
        OUTPUT_RUNS
        / "legacy_realworld_invalid_jsons.csv"
    )

    pd.DataFrame(
        invalid_rows,
        columns=[
            "dataset",
            "phase",
            "path",
            "error",
        ],
    ).to_csv(
        invalid_path,
        index=False,
    )

    print(f"\nSaved: {coverage_path}")
    print(f"Saved: {invalid_path}")

    incomplete = coverage[
        ~coverage["complete_config_grid"]
    ]

    if incomplete.empty:
        print(
            "\nAll legacy configuration grids are complete."
        )
    else:
        print(
            "\nWARNING: incomplete legacy blocks:"
        )
        print(
            incomplete.to_string(index=False)
        )


if __name__ == "__main__":
    main()
