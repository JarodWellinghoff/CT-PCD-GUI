from __future__ import annotations

from pathlib import Path

from ..domain.models import ValidationError


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def validate_output_location(
    output: str | Path,
    *,
    input_locations: tuple[str | Path, ...],
) -> Path:
    """Reject any output that could overwrite or be discovered as an input."""

    target = _resolved(output)
    if not str(output).strip():
        raise ValidationError("Select an output directory.")
    for raw_input in input_locations:
        if not str(raw_input).strip():
            continue
        source = _resolved(raw_input)
        source_root = source if source.is_dir() else source.parent
        if target == source or target == source_root:
            raise ValidationError("The output directory must not be an input directory.")
        if _is_relative_to(target, source_root):
            raise ValidationError(
                "The output directory must not be inside an input tree; choose a separate location."
            )
        if _is_relative_to(source_root, target):
            raise ValidationError(
                "The output directory must not contain an input tree; choose a separate location."
            )
    if target.exists() and not target.is_dir():
        raise ValidationError(f"The output path is not a directory: {target}")
    return target
