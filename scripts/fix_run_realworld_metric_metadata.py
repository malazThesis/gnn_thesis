from __future__ import annotations

import shutil
import textwrap
from pathlib import Path


TARGET = Path("scripts/run_realworld.py")
BACKUP = Path(
    "scripts/run_realworld.py.before_metric_metadata_fix"
)
MARKER = "_metric_scope = dict(locals())"


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip())


def statement_end(
    lines: list[str],
    start: int,
) -> int:
    balance = 0

    for index in range(start, len(lines)):
        line = lines[index]

        balance += line.count("(")
        balance -= line.count(")")
        balance += line.count("[")
        balance -= line.count("]")
        balance += line.count("{")
        balance -= line.count("}")

        if (
            balance <= 0
            and not line.rstrip().endswith("\\")
        ):
            return index

    raise RuntimeError(
        f"Could not find statement end from line "
        f"{start + 1}"
    )


def indent_block(
    block: str,
    indentation: str,
) -> list[str]:
    normalized = textwrap.dedent(block).strip("\n")

    return [
        indentation + line if line else ""
        for line in normalized.splitlines()
    ]


def main():
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    original = TARGET.read_text(
        encoding="utf-8"
    )

    if MARKER in original:
        print(
            "Metric metadata fix is already installed."
        )
        return

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print("Backup:", BACKUP)

    lines = original.splitlines()

    start = next(
        (
            index
            for index, line in enumerate(lines)
            if "_metric_out_dir = MetricsPath(" in line
        ),
        None,
    )

    if start is None:
        raise RuntimeError(
            "_metric_out_dir block was not found."
        )

    csv_start = next(
        (
            index
            for index in range(start, len(lines))
            if "_oversmoothing_csv = (" in lines[index]
        ),
        None,
    )

    if csv_start is None:
        raise RuntimeError(
            "_oversmoothing_csv block was not found."
        )

    end = statement_end(
        lines,
        csv_start,
    )

    indentation = (
        " " * leading_spaces(lines[start])
    )

    replacement = indent_block(
        """
        _metric_scope = dict(locals())

        def _metric_lookup(
            names,
            default=None,
        ):
            for name in names:
                value = _metric_scope.get(name)

                if (
                    value is not None
                    and not isinstance(
                        value,
                        torch.nn.Module,
                    )
                    and isinstance(
                        value,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    )
                ):
                    return value

            for container_name in (
                "config",
                "cfg",
                "row",
                "params",
                "job",
                "task",
                "run_config",
            ):
                source = _metric_scope.get(
                    container_name
                )

                if not hasattr(source, "get"):
                    continue

                for name in names:
                    try:
                        value = source.get(name)
                    except Exception:
                        continue

                    if value is not None:
                        return value

            for name in names:
                value = getattr(
                    args,
                    name,
                    None,
                )

                if value is not None:
                    return value

            return default

        _metric_dataset = str(
            _metric_lookup(
                ("dataset_name", "dataset"),
                "dataset",
            )
        )
        _metric_model = str(
            _metric_lookup(
                (
                    "model_name",
                    "model_type",
                    "model",
                ),
                "model",
            )
        )
        _metric_layers = _metric_lookup(
            ("num_layers", "layers"),
            "x",
        )
        _metric_hidden = _metric_lookup(
            (
                "hidden_channels",
                "hidden_dim",
                "hidden",
            ),
            "x",
        )
        _metric_seed = _metric_lookup(
            ("seed",),
            0,
        )
        _metric_split_idx = _metric_lookup(
            ("split_idx", "split"),
            None,
        )

        _metric_out_dir = MetricsPath(
            getattr(
                args,
                "out_dir",
                "runs",
            )
        )

        _metric_existing_stem = None

        for _metric_candidate in (
            _metric_scope.values()
        ):
            if not isinstance(
                _metric_candidate,
                (str, MetricsPath),
            ):
                continue

            _metric_candidate_name = (
                MetricsPath(
                    _metric_candidate
                ).name
            )

            for _metric_suffix in (
                "_history.csv",
                "_summary.json",
            ):
                if _metric_candidate_name.endswith(
                    _metric_suffix
                ):
                    _metric_existing_stem = (
                        _metric_candidate_name[
                            :-len(_metric_suffix)
                        ]
                    )
                    break

            if _metric_existing_stem is not None:
                break

        if _metric_existing_stem is not None:
            _metric_stem = _metric_existing_stem
        else:
            _metric_stem = (
                _metric_dataset
                + "_"
                + _metric_model
                + "_L"
                + str(_metric_layers)
                + "_H"
                + str(_metric_hidden)
                + "_seed"
                + str(_metric_seed)
            )

            if _metric_split_idx is not None:
                _metric_stem += (
                    "_split"
                    + str(_metric_split_idx)
                )

        _oversmoothing_csv = (
            _metric_out_dir
            / (
                _metric_stem
                + "_oversmoothing.csv"
            )
        )
        """,
        indentation,
    )

    lines[start:end + 1] = replacement

    metadata_start = next(
        (
            index
            for index, line in enumerate(lines)
            if "metadata={" in line
            and index > start
        ),
        None,
    )

    if metadata_start is None:
        raise RuntimeError(
            "collect_layer_metrics metadata block "
            "was not found."
        )

    metadata_indent_count = leading_spaces(
        lines[metadata_start]
    )

    metadata_end = None

    for index in range(
        metadata_start + 1,
        len(lines),
    ):
        if (
            lines[index].strip() == "},"
            and leading_spaces(lines[index])
            == metadata_indent_count
        ):
            metadata_end = index
            break

    if metadata_end is None:
        raise RuntimeError(
            "Could not find metadata block end."
        )

    metadata_indentation = (
        " " * metadata_indent_count
    )

    metadata_replacement = indent_block(
        """
        metadata={
            "dataset": _metric_dataset,
            "model": _metric_model,
            "seed": _metric_seed,
            "split_idx": _metric_split_idx,
            "num_layers": _metric_layers,
            "hidden_channels": _metric_hidden,
        },
        """,
        metadata_indentation,
    )

    lines[
        metadata_start:metadata_end + 1
    ] = metadata_replacement

    patched = "\n".join(lines) + "\n"

    compile(
        patched,
        str(TARGET),
        "exec",
    )

    TARGET.write_text(
        patched,
        encoding="utf-8",
    )

    print("Updated:", TARGET)
    print("Backup:", BACKUP)


if __name__ == "__main__":
    main()
