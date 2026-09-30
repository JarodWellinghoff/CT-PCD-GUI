from __future__ import annotations

from pathlib import Path
from pydicom import dcmread
from pydicom.tag import Tag

from ..application.models import NoiseJobConfig, DicomWorkItem
from ..application.errors import NoiseInsertionCancelled
from ..application.ports import CancellationFlag, LogCallback

SUPPORTED_EXTENSIONS = {".dcm", ".ima"}
TAG_NUMBER_OF_FRAMES = Tag(0x0028, 0x0008)


def _noop(*_args, **_kwargs) -> None:
    return None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _candidate_files(input_path: Path, output_dir: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    # Exclude generated output only when it is a strict child directory of the
    # input tree.  An output directory equal to, or above, the input directory
    # must not hide all source files.
    exclude_output_tree = output_dir != input_path and _path_is_within(
        output_dir, input_path
    )

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not (exclude_output_tree and _path_is_within(path, output_dir))
    ]
    return sorted(files, key=lambda p: str(p).lower())


def _frame_count_from_header(path: Path) -> int:
    ds = dcmread(
        str(path),
        force=True,
        stop_before_pixels=True,
        specific_tags=[TAG_NUMBER_OF_FRAMES],
    )
    value = ds.get(TAG_NUMBER_OF_FRAMES, 1)
    try:
        count = int(value.value if hasattr(value, "value") else value)
    except (TypeError, ValueError):
        count = 1
    return max(1, count)


def _output_path_for(
    source: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    if input_root.is_dir():
        relative = source.relative_to(input_root)
    else:
        relative = Path(source.name)

    extension = source.suffix or ".dcm"
    output_name = f"{source.stem}{suffix}{extension}"
    return output_root / relative.parent / output_name


def discover_work_items(
    config: NoiseJobConfig,
    cancel_event: CancellationFlag,
    log_callback: LogCallback = _noop,
) -> list[DicomWorkItem]:
    """Inspect input headers and build a deterministic processing plan."""

    input_path = Path(config.input_path).expanduser().resolve()
    output_root = Path(config.output_dir).expanduser().resolve()
    candidates = _candidate_files(input_path, output_root, config.recursive)
    if not candidates:
        raise FileNotFoundError(f"No .dcm or .ima files were found at: {input_path}")

    log_callback(f"Inspecting {len(candidates)} DICOM file(s)...")
    items: list[DicomWorkItem] = []

    for index, path in enumerate(candidates, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise NoiseInsertionCancelled("Cancelled while inspecting input files.")

        try:
            # Forced single-frame mode can avoid a second header read for every
            # projection file.  The pixel shape is still validated while processing.
            frame_count = (
                1 if config.input_mode == "single" else _frame_count_from_header(path)
            )
        except Exception as exc:
            message = f"Unable to read DICOM header for {path.name}: {exc}"
            if config.continue_on_error:
                log_callback(f"WARNING: {message}")
                continue
            raise RuntimeError(message) from exc

        detected_multiframe = frame_count > 1
        if config.input_mode == "single" and detected_multiframe:
            message = (
                f"{path.name} contains {frame_count} frames but the selected mode "
                "is Single-frame."
            )
            if config.continue_on_error:
                log_callback(f"WARNING: {message} File skipped.")
                continue
            raise ValueError(message)

        if config.input_mode == "multi" and not detected_multiframe:
            message = (
                f"{path.name} is single-frame but the selected mode is Multi-frame."
            )
            if config.continue_on_error:
                log_callback(f"WARNING: {message} File skipped.")
                continue
            raise ValueError(message)

        is_multiframe = (
            detected_multiframe
            if config.input_mode == "auto"
            else config.input_mode == "multi"
        )
        output_path = _output_path_for(
            path,
            input_path,
            output_root,
            config.file_suffix,
        )
        items.append(
            DicomWorkItem(
                input_path=path,
                output_path=output_path,
                frame_count=frame_count if is_multiframe else 1,
                is_multiframe=is_multiframe,
            )
        )

        if index % 100 == 0:
            log_callback(f"  Inspected {index}/{len(candidates)} files")

    if not items:
        raise RuntimeError("No compatible DICOM files remain after input inspection.")

    return items
