from pathlib import Path

import pytest

from ct_pcd_gui.features.lesion_insertion.domain.models import ValidationError
from ct_pcd_gui.features.lesion_insertion.infrastructure.output_paths import (
    validate_output_location,
)


def test_output_cannot_overwrite_or_nest_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(ValidationError, match="input"):
        validate_output_location(source, input_locations=(source,))
    with pytest.raises(ValidationError, match="inside"):
        validate_output_location(source / "output", input_locations=(source,))


def test_separate_output_is_allowed(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    assert validate_output_location(output, input_locations=(source,)) == output.resolve()


def test_output_cannot_contain_lesion_library(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    with pytest.raises(ValidationError, match="contain an input tree"):
        validate_output_location(tmp_path, input_locations=(library,))
