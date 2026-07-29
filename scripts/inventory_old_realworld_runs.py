from pathlib import Path


RUNS_DIR = Path("runs")

DATASETS = {
    "Cora": ["cora"],
    "CiteSeer": ["citeseer", "cite_seer", "cite-seer"],
    "PubMed": ["pubmed", "pub_med", "pub-med"],
    "Roman-Empire": [
        "roman_empire",
        "roman-empire",
        "romanempire",
    ],
    "Amazon-Photo": [
        "amazon_photo",
        "amazon-photo",
        "amazonphoto",
    ],
}


def normalize(value: str) -> str:
    return (
        value.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def matches(path: Path, aliases: list[str]) -> bool:
    normalized_path = normalize(str(path))

    return any(
        normalize(alias) in normalized_path
        for alias in aliases
    )


def main():
    all_files = [
        path
        for path in RUNS_DIR.rglob("*")
        if path.is_file()
    ]

    for dataset, aliases in DATASETS.items():
        matches_for_dataset = [
            path
            for path in all_files
            if matches(path, aliases)
        ]

        summary_files = [
            path
            for path in matches_for_dataset
            if path.name.endswith("_summary.json")
        ]

        history_files = [
            path
            for path in matches_for_dataset
            if path.name.endswith("_history.csv")
        ]

        grouped_files = [
            path
            for path in matches_for_dataset
            if path.name.endswith("_grouped.csv")
        ]

        best_grouped_files = [
            path
            for path in matches_for_dataset
            if path.name.endswith("_best_grouped.csv")
        ]

        directories = sorted(
            {
                str(path.parent)
                for path in matches_for_dataset
            }
        )

        print("=" * 80)
        print(dataset)
        print("=" * 80)
        print(f"Summary JSON files: {len(summary_files)}")
        print(f"History CSV files:  {len(history_files)}")
        print(f"Grouped CSV files:  {len(grouped_files)}")
        print(f"Best grouped files: {len(best_grouped_files)}")

        print("\nDirectories:")
        if directories:
            for directory in directories:
                print(f"  {directory}")
        else:
            print("  NONE FOUND")

        print("\nGrouped files:")
        if grouped_files:
            for path in sorted(grouped_files):
                print(f"  {path}")
        else:
            print("  NONE FOUND")

        print("\nExamples of summary files:")
        if summary_files:
            for path in sorted(summary_files)[:8]:
                print(f"  {path}")
        else:
            print("  NONE FOUND")

        print()


if __name__ == "__main__":
    main()
