from __future__ import annotations

from pathlib import Path

from ..domain.models import DicomVolume, LesionSession, RawDatasetInfo, ValidationIssue
from ..infrastructure.output_paths import validate_output_location


def validate_session(
    session: LesionSession,
    volume: DicomVolume | None,
    raw_info: RawDatasetInfo | None,
    association_issues: list[ValidationIssue] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = list(association_issues or [])
    try:
        session.validate_basic()
    except Exception as exc:
        issues.append(ValidationIssue("error", "session-invalid", str(exc)))

    if volume is None:
        issues.append(
            ValidationIssue("error", "reconstruction-missing", "Load a reconstructed DICOM series.")
        )
    if raw_info is None:
        issues.append(
            ValidationIssue(
                "error",
                "raw-missing",
                "Load and inspect the associated DICOM-CT-PD data.",
            )
        )
    enabled = [item for item in session.lesions if item.enabled]
    if not enabled:
        issues.append(
            ValidationIssue("error", "lesions-missing", "Add and enable at least one lesion.")
        )
    for lesion in enabled:
        try:
            lesion.validate()
        except Exception as exc:
            issues.append(ValidationIssue("error", "lesion-invalid", str(exc)))

    if volume is not None:
        try:
            volume.geometry.assert_legacy_alpha_compatible()
        except Exception as exc:
            issues.append(ValidationIssue("error", "unsupported-spatial-geometry", str(exc)))

    if raw_info is not None:
        spectra = set(raw_info.spectrum_indices)
        mapping = session.spectrum_map
        missing = sorted(spectra - set(mapping))
        if missing:
            issues.append(
                ValidationIssue(
                    "error",
                    "spectrum-map-incomplete",
                    f"Map CTPD spectrum indices {missing} to lesion-model channels.",
                )
            )
        if mapping and any(spectrum < 1 or channel < 0 for spectrum, channel in mapping.items()):
            issues.append(
                ValidationIssue(
                    "error",
                    "spectrum-map-invalid",
                    "Spectrum indices are 1-based and lesion channels are 0-based.",
                )
            )
        for lesion in enabled:
            invalid_channels = sorted(
                {channel for channel in mapping.values() if channel >= len(lesion.background_hu)}
            )
            if invalid_channels:
                issues.append(
                    ValidationIssue(
                        "error",
                        "lesion-channel-mismatch",
                        f"{lesion.label}: spectrum mapping references lesion channel(s) "
                        f"{invalid_channels}, but this model has "
                        f"{len(lesion.background_hu)} channel(s).",
                    )
                )

    if session.output_directory:
        try:
            validate_output_location(
                session.output_directory,
                input_locations=(
                    session.reconstruction_source,
                    session.raw_source,
                    session.lesion_library_source,
                    *(lesion.lesion_path for lesion in session.lesions),
                ),
            )
        except Exception as exc:
            issues.append(ValidationIssue("error", "output-unsafe", str(exc)))
    else:
        issues.append(ValidationIssue("error", "output-missing", "Select a new output directory."))

    if not session.reconstruction_command.strip():
        issues.append(
            ValidationIssue(
                "warning",
                "reconstruction-not-configured",
                "No reconstruction command is configured. Final processing will produce "
                "modified DICOM-CT-PD data only; the approximate image-domain preview is not "
                "a reconstructed final result.",
            )
        )
    elif not session.reconstruction_output_directory.strip():
        issues.append(
            ValidationIssue(
                "error",
                "reconstruction-output-missing",
                "Select a reconstruction output directory when a reconstruction command is used.",
            )
        )
    else:
        try:
            reconstruction_output = validate_output_location(
                session.reconstruction_output_directory,
                input_locations=(
                    session.reconstruction_source,
                    session.raw_source,
                    session.lesion_library_source,
                    session.output_directory,
                    *(lesion.lesion_path for lesion in session.lesions),
                ),
            )
            if reconstruction_output.exists() and any(reconstruction_output.iterdir()):
                raise ValueError(
                    "The reconstruction output directory is not empty; choose a new location."
                )
        except Exception as exc:
            issues.append(
                ValidationIssue("error", "reconstruction-output-unsafe", str(exc))
            )

    output = Path(session.output_directory).expanduser() if session.output_directory else None
    if output and output.exists() and any(output.iterdir()):
        issues.append(
            ValidationIssue(
                "error",
                "output-not-empty",
                "The selected output directory is not empty. Choose a new or empty directory.",
            )
        )
    return issues
