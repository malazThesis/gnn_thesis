from __future__ import annotations

import re
import shutil
from pathlib import Path


TARGET = Path("scripts/run_realworld.py")
MARKER = "# OVERSMOOTHING_TRACKING_PATCH_V1"


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

        continued = line.rstrip().endswith("\\")

        if balance <= 0 and not continued:
            return index

    raise RuntimeError(
        f"Could not determine statement end at line "
        f"{start + 1}"
    )


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip())


def import_insertion_index(
    lines: list[str],
) -> int:
    index = 0

    if lines and lines[0].startswith("#!"):
        index += 1

    while (
        index < len(lines)
        and (
            not lines[index].strip()
            or "coding:" in lines[index]
        )
    ):
        index += 1

    if index < len(lines):
        stripped = lines[index].lstrip()

        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]

            if stripped.count(quote) >= 2:
                index += 1
            else:
                index += 1

                while index < len(lines):
                    if quote in lines[index]:
                        index += 1
                        break
                    index += 1

    while (
        index < len(lines)
        and lines[index].startswith(
            "from __future__ import"
        )
    ):
        index += 1

    return index


def detect_model_call(text: str):
    patterns = [
        (
            r"model\s*\(\s*data\.x\s*,"
            r"\s*data\.edge_index\s*\)",
            "model(data.x, data.edge_index)",
            "data.edge_index",
        ),
        (
            r"model\s*\(\s*x\s*,"
            r"\s*edge_index\s*\)",
            "model(x, edge_index)",
            "edge_index",
        ),
    ]

    for pattern, call, edge_index in patterns:
        if re.search(pattern, text, flags=re.S):
            return call, edge_index

    raise RuntimeError(
        "Could not safely detect the model forward call. "
        "Expected model(data.x, data.edge_index) "
        "or model(x, edge_index)."
    )


def main():
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    original = TARGET.read_text(
        encoding="utf-8"
    )

    if MARKER in original:
        print(
            "run_realworld.py is already patched."
        )
        return

    backup = TARGET.with_name(
        "run_realworld.py.before_oversmoothing"
    )

    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print("Backup:", backup)

    model_call, edge_index_expression = (
        detect_model_call(original)
    )

    lines = original.splitlines()

    # --------------------------------------------------
    # Imports
    # --------------------------------------------------
    import_index = import_insertion_index(lines)

    import_block = [
        "",
        MARKER,
        "import atexit",
        "from pathlib import Path as MetricsPath",
        "",
        "from analysis_metrics.oversmoothing import (",
        "    LayerEmbeddingRecorder,",
        "    append_metrics_csv,",
        "    collect_layer_metrics,",
        "    default_metric_epochs,",
        "    model_gradient_norm,",
        ")",
        "",
    ]

    lines[
        import_index:import_index
    ] = import_block

    # --------------------------------------------------
    # argparse options
    # --------------------------------------------------
    parse_index = None
    parser_variable = None
    args_variable = None

    parse_pattern = re.compile(
        r"^\s*(?P<args>[A-Za-z_]\w*)\s*="
        r"\s*(?P<parser>[A-Za-z_]\w*)"
        r"\.parse_args\s*\("
    )

    for index, line in enumerate(lines):
        match = parse_pattern.search(line)

        if match:
            parse_index = index
            parser_variable = match.group("parser")
            args_variable = match.group("args")
            break

    if parse_index is None:
        raise RuntimeError(
            "Could not find parser.parse_args()."
        )

    argparse_indent = re.match(
        r"^\s*",
        lines[parse_index],
    ).group(0)

    argument_block = [
        (
            f'{argparse_indent}{parser_variable}.add_argument('
        ),
        (
            f'{argparse_indent}    '
            '"--track-oversmoothing",'
        ),
        (
            f'{argparse_indent}    '
            'action="store_true",'
        ),
        (
            f'{argparse_indent}    '
            'help="Record layerwise oversmoothing metrics.",'
        ),
        f"{argparse_indent})",
        (
            f'{argparse_indent}{parser_variable}.add_argument('
        ),
        (
            f'{argparse_indent}    '
            '"--oversmoothing-epochs",'
        ),
        (
            f'{argparse_indent}    '
            'type=str,'
        ),
        (
            f'{argparse_indent}    '
            'default="",'
        ),
        (
            f'{argparse_indent}    '
            'help="Comma-separated epochs to record.",'
        ),
        f"{argparse_indent})",
        (
            f'{argparse_indent}{parser_variable}.add_argument('
        ),
        (
            f'{argparse_indent}    '
            '"--oversmoothing-max-nodes",'
        ),
        (
            f'{argparse_indent}    '
            'type=int,'
        ),
        (
            f'{argparse_indent}    '
            'default=2048,'
        ),
        f"{argparse_indent})",
        (
            f'{argparse_indent}{parser_variable}.add_argument('
        ),
        (
            f'{argparse_indent}    '
            '"--oversmoothing-max-edges",'
        ),
        (
            f'{argparse_indent}    '
            'type=int,'
        ),
        (
            f'{argparse_indent}    '
            'default=200000,'
        ),
        f"{argparse_indent})",
        "",
    ]

    lines[
        parse_index:parse_index
    ] = argument_block

    # Find parse_args again after insertion.
    for index, line in enumerate(lines):
        if parse_pattern.search(line):
            parse_index = index
            break

    parse_end = statement_end(
        lines,
        parse_index,
    )

    setup_indent = re.match(
        r"^\s*",
        lines[parse_index],
    ).group(0)

    args_setup = [
        "",
        (
            f"{setup_indent}"
            "_oversmoothing_epochs = ("
        ),
        (
            f"{setup_indent}    "
            "{"
        ),
        (
            f"{setup_indent}        "
            "int(value.strip())"
        ),
        (
            f"{setup_indent}        "
            f"for value in "
            f"{args_variable}.oversmoothing_epochs"
            ".split(',')"
        ),
        (
            f"{setup_indent}        "
            "if value.strip()"
        ),
        (
            f"{setup_indent}    "
            "}"
        ),
        (
            f"{setup_indent}    "
            f"if {args_variable}."
            "oversmoothing_epochs.strip()"
        ),
        (
            f"{setup_indent}    "
            "else default_metric_epochs("
        ),
        (
            f"{setup_indent}        "
            "int(getattr("
            f"{args_variable}, "
            '"epochs", 200'
            "))"
        ),
        (
            f"{setup_indent}    "
            ")"
        ),
        f"{setup_indent})",
        "",
    ]

    lines[
        parse_end + 1:parse_end + 1
    ] = args_setup

    # --------------------------------------------------
    # Locate outer epoch loop.
    # --------------------------------------------------
    epoch_index = None
    epoch_variable = None

    epoch_pattern = re.compile(
        r"^(?P<indent>\s*)for\s+"
        r"(?P<epoch>[A-Za-z_]\w*)\s+in\s+"
    )

    for index, line in enumerate(lines):
        match = epoch_pattern.search(line)

        if match and "epoch" in match.group(
            "epoch"
        ).lower():
            epoch_index = index
            epoch_variable = match.group("epoch")
            break

    if epoch_index is None:
        raise RuntimeError(
            "Could not locate the epoch loop."
        )

    epoch_line = lines[epoch_index]
    epoch_indent = leading_spaces(epoch_line)

    zero_based = bool(
        re.search(
            r"range\s*\(\s*"
            r"(?:[A-Za-z_]\w*\.)?epochs"
            r"\s*\)",
            epoch_line,
        )
    )

    if zero_based:
        metric_epoch_expression = (
            f"int({epoch_variable}) + 1"
        )
    else:
        metric_epoch_expression = (
            f"int({epoch_variable})"
        )

    # --------------------------------------------------
    # Recorder initialization after optimizer creation.
    # --------------------------------------------------
    optimizer_index = None

    optimizer_pattern = re.compile(
        r"^\s*optimizer\s*="
    )

    for index, line in enumerate(lines):
        if (
            index < epoch_index
            and optimizer_pattern.search(line)
        ):
            optimizer_index = index

    if optimizer_index is None:
        raise RuntimeError(
            "Could not locate optimizer creation "
            "before the epoch loop."
        )

    optimizer_end = statement_end(
        lines,
        optimizer_index,
    )

    optimizer_indent = re.match(
        r"^\s*",
        lines[optimizer_index],
    ).group(0)

    recorder_block = [
        "",
        (
            f"{optimizer_indent}"
            "_oversmoothing_recorder = ("
        ),
        (
            f"{optimizer_indent}    "
            "LayerEmbeddingRecorder(model)"
        ),
        (
            f"{optimizer_indent}    "
            f"if {args_variable}."
            "track_oversmoothing"
        ),
        (
            f"{optimizer_indent}    "
            "else None"
        ),
        f"{optimizer_indent})",
        (
            f"{optimizer_indent}"
            "if _oversmoothing_recorder "
            "is not None:"
        ),
        (
            f"{optimizer_indent}    "
            "atexit.register("
        ),
        (
            f"{optimizer_indent}        "
            "_oversmoothing_recorder.close"
        ),
        (
            f"{optimizer_indent}    "
            ")"
        ),
        "",
        (
            f"{optimizer_indent}"
            "_metric_out_dir = MetricsPath("
        ),
        (
            f"{optimizer_indent}    "
            f"getattr({args_variable}, "
            '"out_dir", "runs")'
        ),
        f"{optimizer_indent})",
        (
            f"{optimizer_indent}"
            "_metric_split_idx = getattr("
        ),
        (
            f"{optimizer_indent}    "
            f"{args_variable}, "
            '"split_idx", None'
        ),
        f"{optimizer_indent})",
        (
            f"{optimizer_indent}"
            "_metric_stem = ("
        ),
        (
            f"{optimizer_indent}    "
            'f"{getattr('
            f"{args_variable}, "
            "'dataset', 'dataset'"
            ")}_"
        ),
        (
            f"{optimizer_indent}    "
            'f"{getattr('
            f"{args_variable}, "
            "'model', "
            "getattr("
            f"{args_variable}, "
            "'model_name', 'model'"
            ")"
            ")}_"
        ),
        (
            f"{optimizer_indent}    "
            'f"L{getattr('
            f"{args_variable}, "
            "'num_layers', "
            "getattr("
            f"{args_variable}, "
            "'layers', 'x'"
            ")"
            ")}_"
        ),
        (
            f"{optimizer_indent}    "
            'f"H{getattr('
            f"{args_variable}, "
            "'hidden_channels', "
            "getattr("
            f"{args_variable}, "
            "'hidden', 'x'"
            ")"
            ")}_"
        ),
        (
            f"{optimizer_indent}    "
            'f"seed{getattr('
            f"{args_variable}, "
            "'seed', 0"
            ')}"'
        ),
        f"{optimizer_indent})",
        (
            f"{optimizer_indent}"
            "if _metric_split_idx is not None:"
        ),
        (
            f"{optimizer_indent}    "
            "_metric_stem += "
            'f"_split{_metric_split_idx}"'
        ),
        (
            f"{optimizer_indent}"
            "_oversmoothing_csv = ("
        ),
        (
            f"{optimizer_indent}    "
            "_metric_out_dir"
        ),
        (
            f"{optimizer_indent}    "
            '/ f"{_metric_stem}_'
            'oversmoothing.csv"'
        ),
        f"{optimizer_indent})",
        "",
    ]

    lines[
        optimizer_end + 1:optimizer_end + 1
    ] = recorder_block

    # Re-find epoch loop after insertion.
    for index, line in enumerate(lines):
        match = epoch_pattern.search(line)

        if (
            match
            and match.group("epoch")
            == epoch_variable
        ):
            epoch_index = index
            epoch_indent = leading_spaces(line)
            break

    # --------------------------------------------------
    # Insert recording after train(...) in epoch loop.
    # --------------------------------------------------
    train_call_index = None
    loop_end = len(lines)

    for index in range(
        epoch_index + 1,
        len(lines),
    ):
        line = lines[index]

        if (
            line.strip()
            and leading_spaces(line) <= epoch_indent
        ):
            loop_end = index
            break

        if (
            re.search(r"\btrain\s*\(", line)
            and not line.lstrip().startswith(
                "def train"
            )
        ):
            train_call_index = index
            break

    if train_call_index is None:
        for index in range(
            epoch_index + 1,
            loop_end,
        ):
            if "optimizer.step(" in lines[index]:
                train_call_index = index
                break

    if train_call_index is None:
        raise RuntimeError(
            "Could not locate a training call "
            "inside the epoch loop."
        )

    train_call_end = statement_end(
        lines,
        train_call_index,
    )

    block_indent = re.match(
        r"^\s*",
        lines[train_call_index],
    ).group(0)

    record_block = [
        "",
        (
            f"{block_indent}"
            f"_metric_epoch = "
            f"{metric_epoch_expression}"
        ),
        (
            f"{block_indent}"
            "if ("
        ),
        (
            f"{block_indent}    "
            "_oversmoothing_recorder "
            "is not None"
        ),
        (
            f"{block_indent}    "
            "and _metric_epoch "
            "in _oversmoothing_epochs"
        ),
        (
            f"{block_indent}"
            "):"
        ),
        (
            f"{block_indent}    "
            "model.eval()"
        ),
        (
            f"{block_indent}    "
            "_oversmoothing_recorder.start()"
        ),
        (
            f"{block_indent}    "
            "with torch.no_grad():"
        ),
        (
            f"{block_indent}        "
            f"_ = {model_call}"
        ),
        (
            f"{block_indent}    "
            "_oversmoothing_recorder.stop()"
        ),
        "",
        (
            f"{block_indent}    "
            "_metric_rows = "
            "collect_layer_metrics("
        ),
        (
            f"{block_indent}        "
            "embeddings="
            "_oversmoothing_recorder.embeddings,"
        ),
        (
            f"{block_indent}        "
            f"edge_index="
            f"{edge_index_expression},"
        ),
        (
            f"{block_indent}        "
            "metadata={"
        ),
        (
            f"{block_indent}            "
            '"dataset": getattr('
            f"{args_variable}, "
            '"dataset", None),'
        ),
        (
            f"{block_indent}            "
            '"model": getattr('
            f"{args_variable}, "
            '"model", getattr('
            f"{args_variable}, "
            '"model_name", None)),'
        ),
        (
            f"{block_indent}            "
            '"seed": getattr('
            f"{args_variable}, "
            '"seed", None),'
        ),
        (
            f"{block_indent}            "
            '"split_idx": getattr('
            f"{args_variable}, "
            '"split_idx", None),'
        ),
        (
            f"{block_indent}            "
            '"num_layers": getattr('
            f"{args_variable}, "
            '"num_layers", getattr('
            f"{args_variable}, "
            '"layers", None)),'
        ),
        (
            f"{block_indent}            "
            '"hidden_channels": getattr('
            f"{args_variable}, "
            '"hidden_channels", getattr('
            f"{args_variable}, "
            '"hidden", None)),'
        ),
        (
            f"{block_indent}        "
            "},"
        ),
        (
            f"{block_indent}        "
            "epoch=_metric_epoch,"
        ),
        (
            f"{block_indent}        "
            "gradient_norm="
            "model_gradient_norm(model),"
        ),
        (
            f"{block_indent}        "
            "max_pairwise_nodes="
            f"{args_variable}."
            "oversmoothing_max_nodes,"
        ),
        (
            f"{block_indent}        "
            "max_rank_nodes="
            f"{args_variable}."
            "oversmoothing_max_nodes,"
        ),
        (
            f"{block_indent}        "
            "max_edges="
            f"{args_variable}."
            "oversmoothing_max_edges,"
        ),
        (
            f"{block_indent}    "
            ")"
        ),
        (
            f"{block_indent}    "
            "append_metrics_csv("
        ),
        (
            f"{block_indent}        "
            "_oversmoothing_csv,"
        ),
        (
            f"{block_indent}        "
            "_metric_rows,"
        ),
        (
            f"{block_indent}    "
            ")"
        ),
        "",
    ]

    lines[
        train_call_end + 1:train_call_end + 1
    ] = record_block

    patched = "\n".join(lines) + "\n"

    # REPAIR_GENERATED_METRIC_STEM_V1
    def repair_metric_stem(match):
        indent = match.group("indent")

        return (
            f"{indent}_metric_dataset = str(getattr("
            f"{args_variable}, 'dataset', 'dataset'))\n"
            f"{indent}_metric_model = str(getattr("
            f"{args_variable}, 'model', getattr("
            f"{args_variable}, 'model_name', 'model')))\n"
            f"{indent}_metric_layers = getattr("
            f"{args_variable}, 'num_layers', getattr("
            f"{args_variable}, 'layers', 'x'))\n"
            f"{indent}_metric_hidden = getattr("
            f"{args_variable}, 'hidden_channels', getattr("
            f"{args_variable}, 'hidden', 'x'))\n"
            f"{indent}_metric_seed = getattr("
            f"{args_variable}, 'seed', 0)\n"
            f"{indent}_metric_stem = (\n"
            f"{indent}    _metric_dataset\n"
            f"{indent}    + '_' + _metric_model\n"
            f"{indent}    + '_L' + str(_metric_layers)\n"
            f"{indent}    + '_H' + str(_metric_hidden)\n"
            f"{indent}    + '_seed' + str(_metric_seed)\n"
            f"{indent})\n"
            f"{indent}if _metric_split_idx is not None:\n"
            f"{indent}    _metric_stem += (\n"
            f"{indent}        '_split'\n"
            f"{indent}        + str(_metric_split_idx)\n"
            f"{indent}    )\n"
            f"{indent}_oversmoothing_csv = ("
        )

    patched, repair_count = re.subn(
        (
            r"(?ms)^"
            r"(?P<indent>[ \t]*)"
            r"_metric_stem = \(\n"
            r".*?"
            r"^(?P=indent)"
            r"_oversmoothing_csv = \("
        ),
        repair_metric_stem,
        patched,
        count=1,
    )

    if repair_count != 1:
        raise RuntimeError(
            "Could not repair the generated "
            "metric filename block."
        )

    compile(
        patched,
        str(TARGET),
        "exec",
    )

    TARGET.write_text(
        patched,
        encoding="utf-8",
    )

    print("Patched:", TARGET)
    print("Forward call:", model_call)
    print(
        "Epoch expression:",
        metric_epoch_expression,
    )
    print("Backup:", backup)


if __name__ == "__main__":
    main()
