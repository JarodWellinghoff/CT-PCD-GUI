from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lesion_pipeline.coordinates import pixel_center_to_ctpd


def _series(patient_position: str):
    header = SimpleNamespace(
        ReconstructionDiameter=400.0,
        ReconstructionTargetCenterPatient=[10.0, 20.0, 0.0],
        DataCollectionCenterPatient=[3.0, 5.0, 0.0],
        PatientPosition=patient_position,
    )
    return SimpleNamespace(
        first_header=header,
        shape=(4, 400, 400),
        column_spacing_mm=1.0,
        row_spacing_mm=1.0,
        slice_spacing_mm=2.0,
        slice_locations=np.asarray([10.0, 12.0, 14.0, 16.0]),
    )


def test_hfs_center_conversion_matches_matlab_hg_v2():
    converted = pixel_center_to_ctpd([210.0, 220.0, 2.0], _series("HFS"))
    # corrected ImageJ mm = [203, 205, 4]
    assert np.allclose(converted, [3.0, -5.0, 14.0])


def test_ffs_flips_x_and_slice_location():
    converted = pixel_center_to_ctpd([210.0, 220.0, 2.0], _series("FFS"))
    assert np.allclose(converted, [-3.0, -5.0, -14.0])
