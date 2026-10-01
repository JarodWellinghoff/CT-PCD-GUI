from __future__ import annotations

import numpy as np

from .dicom_series import ReconSeries


def pixel_center_to_ctpd(
    center_pixel: list[float] | tuple[float, float, float] | np.ndarray,
    series: ReconSeries,
) -> np.ndarray:
    """Convert GUI/ImageJ (column, row, slice) coordinates to CT-PD x/y/z.

    This preserves the coordinate conversion in the validated MATLAB Alpha
    workflow, including its HFS/FFS orientation convention. The slice index is
    zero-based in Python.
    """

    column, row, slice_index = np.asarray(center_pixel, dtype=float)
    header = series.first_header
    reconstruction_diameter = float(
        getattr(header, "ReconstructionDiameter", series.shape[2] * series.column_spacing_mm)
    )
    imagej_mm = np.asarray(
        [
            column * series.column_spacing_mm,
            row * series.row_spacing_mm,
            slice_index * series.slice_spacing_mm,
        ],
        dtype=float,
    )
    data_center = np.asarray(
        getattr(header, "DataCollectionCenterPatient", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    recon_center = np.asarray(
        getattr(header, "ReconstructionTargetCenterPatient", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    # Match lesion_insertion_alpha_HG_v2.m, the routine called by the previous
    # Alpha workflow: selected ImageJ position - reconstruction target center
    # + data-collection center.
    corrected = imagej_mm - recon_center + data_center

    patient_position = str(getattr(header, "PatientPosition", "HFS")).upper()
    if patient_position == "HFS":
        x_value = corrected[0] - reconstruction_diameter / 2.0
        y_value = -(corrected[1] - reconstruction_diameter / 2.0)
    elif patient_position == "FFS":
        x_value = -(corrected[0] - reconstruction_diameter / 2.0)
        y_value = -(corrected[1] - reconstruction_diameter / 2.0)
    else:
        raise ValueError(
            f"Patient position {patient_position!r} is not supported; use HFS or FFS."
        )

    mapped_index = int(np.rint(corrected[2] / series.slice_spacing_mm))
    mapped_index = int(np.clip(mapped_index, 0, len(series.slice_locations) - 1))
    z_value = float(series.slice_locations[mapped_index])
    if patient_position == "FFS":
        z_value = -z_value
    return np.asarray([x_value, y_value, z_value], dtype=float)
