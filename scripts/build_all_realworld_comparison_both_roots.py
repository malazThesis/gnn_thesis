import os
from pathlib import Path

import build_all_realworld_comparison as base


USER = os.environ["USER"]

RUNS_DIRS = [
    Path(f"/work/log1/{USER}/gnn_thesis_cluster/runs"),
    Path(f"/u/{USER}/gnn_thesis_cluster/runs"),
]


def merge_candidates():
    merged = {}

    for runs_dir in RUNS_DIRS:
        print(f"Searching: {runs_dir}")

        if not runs_dir.exists():
            print("  Directory does not exist — skipped.")
            continue

        base.RUNS_DIR = runs_dir
        current = base.discover_grouped_files()

        print(
            f"  Found {len(current)} dataset/block candidates."
        )

        for key, candidate in current.items():
            previous = merged.get(key)

            if previous is None:
                merged[key] = candidate
                continue

            new_rank = (
                candidate["num_rows"],
                candidate["mtime"],
            )
            old_rank = (
                previous["num_rows"],
                previous["mtime"],
            )

            if new_rank > old_rank:
                merged[key] = candidate

    return merged


def main():
    candidates = merge_candidates()

    print(
        f"\nMerged dataset/block candidates: "
        f"{len(candidates)}"
    )

    best_df = base.build_best_config_table(candidates)
    best_df.to_csv(base.OUTPUT_BEST, index=False)

    comparison, incomplete_df = base.build_comparison(
        best_df
    )

    comparison.to_csv(
        base.OUTPUT_COMPARISON,
        index=False,
    )
    incomplete_df.to_csv(
        base.OUTPUT_INCOMPLETE,
        index=False,
    )

    (
        pairnorm_df,
        matched_detail_df,
        matched_summary_df,
    ) = base.build_pairnorm_tables(candidates)

    pairnorm_df.to_csv(
        base.OUTPUT_PAIRNORM,
        index=False,
    )
    matched_detail_df.to_csv(
        base.OUTPUT_MATCHED_DETAIL,
        index=False,
    )
    matched_summary_df.to_csv(
        base.OUTPUT_MATCHED_SUMMARY,
        index=False,
    )

    print(f"\nSaved: {base.OUTPUT_BEST}")
    print(f"Saved: {base.OUTPUT_COMPARISON}")
    print(f"Saved: {base.OUTPUT_PAIRNORM}")
    print(f"Saved: {base.OUTPUT_MATCHED_DETAIL}")
    print(f"Saved: {base.OUTPUT_MATCHED_SUMMARY}")
    print(f"Saved: {base.OUTPUT_INCOMPLETE}")

    base.print_summary(
        comparison,
        pairnorm_df,
        incomplete_df,
    )


if __name__ == "__main__":
    main()
