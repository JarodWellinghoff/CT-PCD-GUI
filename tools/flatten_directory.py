#!/usr/bin/env python3
"""
Copy files from a directory tree into a flat, batched output structure.

Examples
--------
Create a separate flat output with the source directory name prefixed to
all copied filenames. Files are divided into batches of 20 by default:

    python flatten_directory.py CT-PCD-GUI \
        --output CT-PCD-GUI-flat \
        --include-parent \
        --ignore-dir .git \
        --ignore-dir .venv \
        --ignore-dir __pycache__

Use batches of 100 files instead:

    python flatten_directory.py CT-PCD-GUI \
        --output CT-PCD-GUI-flat \
        --include-parent \
        --batch-size 100

Copy into batch directories inside the source directory:

    python flatten_directory.py CT-PCD-GUI \
        --in-place \
        --include-parent \
        --batch-size 20

A source file such as:

    CT-PCD-GUI/src/ct_pcd_gui/bootstrap.py

becomes:

    CT-PCD-GUI-flat/batch_001/CT-PCD-GUI.src.ct_pcd_gui.bootstrap.py

The source tree is never deleted or modified. With --in-place, generated
batch_NNN directories are placed inside the source directory and are skipped
on subsequent runs.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

BATCH_DIRECTORY_PREFIX = "batch_"
MINIMUM_BATCH_NUMBER_WIDTH = 3
DEFAULT_BATCH_SIZE = 20


@dataclass(frozen=True)
class CopyOperation:
    source: Path
    destination: Path


def is_within(path: Path, parent: Path) -> bool:
    """Return True when path is parent or one of parent's descendants."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_patterns(patterns: Sequence[str]) -> tuple[str, ...]:
    """Normalize ignore patterns to forward-slash relative paths."""
    normalized: list[str] = []

    for pattern in patterns:
        pattern = pattern.strip().replace("\\", "/").strip("/")
        if pattern:
            normalized.append(pattern)

    return tuple(normalized)


def directory_is_ignored(
    relative_directory: Path,
    patterns: Sequence[str],
) -> bool:
    """
    Check whether a relative directory matches an ignore pattern.

    Patterns without a slash match directory names anywhere in the tree:

        .git
        __pycache__
        build*

    Patterns containing a slash match paths relative to the source:

        src/generated
        tests/fixtures/*
    """
    relative_path = relative_directory.as_posix()
    directory_name = relative_directory.name

    for pattern in patterns:
        # Match a directory name at any level.
        if fnmatch.fnmatchcase(directory_name, pattern):
            return True

        # Match a path relative to the source directory.
        if fnmatch.fnmatchcase(relative_path, pattern):
            return True

        # Allow **/name to also match name at the source root.
        if pattern.startswith("**/"):
            if fnmatch.fnmatchcase(relative_path, pattern[3:]):
                return True

    return False


def is_generated_batch_directory(directory_name: str) -> bool:
    """Return True for names such as batch_001 or batch_42."""
    if not directory_name.startswith(BATCH_DIRECTORY_PREFIX):
        return False

    numeric_suffix = directory_name[len(BATCH_DIRECTORY_PREFIX) :]
    return bool(numeric_suffix) and numeric_suffix.isdigit()


def raise_walk_error(error: OSError) -> None:
    """Cause os.walk errors to be reported instead of silently ignored."""
    raise error


def collect_source_files(
    source: Path,
    output: Path,
    ignored_directories: Sequence[str],
) -> list[Path]:
    """Collect files from source while pruning ignored/output directories."""
    files: list[Path] = []

    # Do not recursively process the output when it is inside the source.
    output_is_source_subdirectory = output != source and is_within(output, source)
    in_place = output == source

    for root, directory_names, file_names in os.walk(
        source,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        root_path = Path(root)
        retained_directories: list[str] = []

        for directory_name in directory_names:
            directory = root_path / directory_name
            relative_directory = directory.relative_to(source)

            if output_is_source_subdirectory and directory.resolve() == output:
                continue

            # In --in-place mode, do not treat previously generated batches as
            # new source content. Only source-root batch directories are pruned.
            if (
                in_place
                and root_path == source
                and is_generated_batch_directory(directory_name)
            ):
                continue

            if directory_is_ignored(
                relative_directory,
                ignored_directories,
            ):
                continue

            retained_directories.append(directory_name)

        # Prune ignored/output directories from os.walk.
        directory_names[:] = retained_directories

        for file_name in file_names:
            files.append(root_path / file_name)

    files.sort(key=lambda path: path.relative_to(source).as_posix())
    return files


def batch_directory_name(batch_number: int, number_width: int) -> str:
    """Construct a deterministic batch directory name."""
    return f"{BATCH_DIRECTORY_PREFIX}{batch_number:0{number_width}d}"


def build_copy_plan(
    source: Path,
    output: Path,
    ignored_directories: Sequence[str],
    separator: str,
    overwrite: bool,
    include_parent: bool,
    batch_size: int,
) -> list[CopyOperation]:
    """Build and validate the complete copy plan before copying files."""
    if not separator:
        raise ValueError("The filename separator cannot be empty.")

    if "/" in separator or "\\" in separator:
        raise ValueError("The filename separator cannot contain '/' or '\\'.")

    if batch_size <= 0:
        raise ValueError("The batch size must be greater than zero.")

    if include_parent and not source.name:
        raise ValueError(
            "The source directory has no usable name for --include-parent."
        )

    ignored_directories = normalize_patterns(ignored_directories)
    source_files = collect_source_files(
        source,
        output,
        ignored_directories,
    )

    total_batches = (
        (len(source_files) + batch_size - 1) // batch_size if source_files else 0
    )
    batch_number_width = max(
        MINIMUM_BATCH_NUMBER_WIDTH,
        len(str(total_batches)),
    )

    operations: list[CopyOperation] = []
    flattened_names: dict[str, Path] = {}
    destinations: dict[str, Path] = {}

    for file_index, source_file in enumerate(source_files):
        relative_path = source_file.relative_to(source)
        flattened_parts = list(relative_path.parts)

        if include_parent:
            flattened_parts.insert(0, source.name)

        flattened_name = separator.join(flattened_parts)

        # Detect flattening collisions globally, even when the colliding files
        # would otherwise land in different batch directories.
        flattened_name_key = os.path.normcase(flattened_name)
        previous_source = flattened_names.get(flattened_name_key)
        if previous_source is not None and previous_source != source_file:
            raise RuntimeError(
                "Flattened filename collision detected:\n"
                f"  {previous_source}\n"
                f"  {source_file}\n"
                "Both would use the flattened filename:\n"
                f"  {flattened_name}"
            )

        flattened_names[flattened_name_key] = source_file

        batch_number = (file_index // batch_size) + 1
        batch_name = batch_directory_name(batch_number, batch_number_width)
        batch_directory = output / batch_name
        destination = batch_directory / flattened_name

        if batch_directory.exists() and not batch_directory.is_dir():
            raise NotADirectoryError(
                f"Batch path exists but is not a directory: {batch_directory}"
            )

        # normcase handles case-insensitive destination collisions on Windows.
        destination_key = os.path.normcase(str(destination))
        previous_destination_source = destinations.get(destination_key)
        if (
            previous_destination_source is not None
            and previous_destination_source != source_file
        ):
            raise RuntimeError(
                "Destination collision detected:\n"
                f"  {previous_destination_source}\n"
                f"  {source_file}\n"
                "Both would be copied to:\n"
                f"  {destination}"
            )

        destinations[destination_key] = source_file

        if destination.exists():
            if destination.is_dir():
                raise IsADirectoryError(f"Destination is a directory: {destination}")

            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {destination}\n"
                    "Use --overwrite to replace existing destination files."
                )

        operations.append(
            CopyOperation(
                source=source_file,
                destination=destination,
            )
        )

    return operations


def flatten_directory(
    source: Path,
    output: Path,
    ignored_directories: Sequence[str] = (),
    separator: str = ".",
    overwrite: bool = False,
    dry_run: bool = False,
    include_parent: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Flatten source into batched output and return the copied file count."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source}")

    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output}")

    operations = build_copy_plan(
        source=source,
        output=output,
        ignored_directories=ignored_directories,
        separator=separator,
        overwrite=overwrite,
        include_parent=include_parent,
        batch_size=batch_size,
    )

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    action = "WOULD COPY" if dry_run else "COPY"

    for operation in operations:
        print(f"{action}: {operation.source} -> {operation.destination}")

        if not dry_run:
            operation.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                operation.source,
                operation.destination,
            )

    return len(operations)


def positive_integer(value: str) -> int:
    """Argparse type requiring an integer greater than zero."""
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, received: {value!r}"
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed_value


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a directory tree into flat filenames organized into "
            "fixed-size batch directories."
        )
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Source directory to flatten.",
    )

    output_group = parser.add_mutually_exclusive_group()

    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output directory. By default, a sibling directory named "
            "<source>_flat is used."
        ),
    )

    output_group.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Create batch directories inside the source directory. "
            "Original files and directories are not deleted."
        ),
    )

    parser.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Directory name or relative-path glob to ignore. "
            "Repeat this option for multiple patterns."
        ),
    )

    parser.add_argument(
        "--include-parent",
        action="store_true",
        help=(
            "Prefix every flattened filename with the source directory name. "
            "For example, src.module.py becomes PROJECT.src.module.py."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=(
            "Maximum number of copied files in each batch directory. "
            f"Default: {DEFAULT_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--separator",
        default=".",
        help="Separator used between path components. Default: '.'",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed operations without copying files.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    source = args.source.expanduser().resolve()

    if args.in_place:
        output = source
    elif args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        if not source.name:
            print(
                "error: an output directory must be specified when "
                "the source is a filesystem root",
                file=sys.stderr,
            )
            return 1

        output = source.parent / f"{source.name}_flat"

    try:
        file_count = flatten_directory(
            source=source,
            output=output,
            ignored_directories=args.ignore_dir,
            separator=args.separator,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            include_parent=args.include_parent,
            batch_size=args.batch_size,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    batch_count = (
        (file_count + args.batch_size - 1) // args.batch_size if file_count else 0
    )
    result = "would be copied" if args.dry_run else "copied"
    batch_result = "would be used" if args.dry_run else "used"

    print(
        f"\n{file_count} file(s) {result} to: {output}\n"
        f"{batch_count} batch folder(s) {batch_result}; "
        f"batch size: {args.batch_size}."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
