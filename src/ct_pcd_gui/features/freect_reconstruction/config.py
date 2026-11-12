from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

SECTIONS = ("RawInput", "ScanGeometry", "Preprocessing", "ReconVolume", "ReconSpecific", "ImageOutput")
FIELDS = {
    "RawInput": {"Battelle", "Binary", "Ptr", "Ctd"},
    "ScanGeometry": set("Manufacturer DetectorRows DetectorChannels DetectorTransverseSpacing "
                        "DetectorAxialSpacing DetectorShape DistSourceToDetector DistSourceToIsocenter "
                        "DetectorCentralRow DetectorCentralChannel ProjectionGeometry ScanType "
                        "ProjectionsPerRotation FlyingFocalSpotMode StartTablePosition EndTablePosition "
                        "TableFeedPerRotation AcquisitionFieldOfView CollimatedSliceWidth "
                        "HUCalibrationFactor NumProjectionsTotal".split()),
    "Preprocessing": {"CorrectMATLABIndexing", "FlipDetectorChannels", "FlipDetectorRows",
                      "NegateTubeAngle", "AdapativeFiltration"},
    "ReconVolume": {"StartPosition", "EndPosition", "FieldOfView", "SliceThickness", "SlicePitch",
                    "Nx", "Ny", "XOrigin", "YOrigin"},
    "ReconSpecific": {"KernelA", "KernelC"},
    "ImageOutput": {"RescaleToHU", "AngleOffset", "Binary", "DICOM", "NIfTI1", "Png"},
}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Invalid or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(text: str) -> dict:
    if len(text) > 262144:
        raise ValueError("FreeCT configuration is unexpectedly large (limit: 256 KiB).")
    try:
        data = yaml.load(text, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid FreeCT YAML: {exc}") from exc
    if not isinstance(data, dict) or set(data) != set(SECTIONS):
        raise ValueError(f"Expected exactly these FreeCT v2 sections: {', '.join(SECTIONS)}.")
    for section in SECTIONS:
        if not isinstance(data[section], dict):
            raise ValueError(f"{section} must be a mapping.")
        unknown = set(data[section]) - FIELDS[section]
        if unknown:
            raise ValueError(f"Unknown {section} fields: {', '.join(sorted(unknown))}.")
    return data


def dump_config(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def number(section: dict, key: str, *, positive: bool = False, integer: bool = False):
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{key} must be greater than zero.")
    if integer and (type(value) is not int):
        raise ValueError(f"{key} must be a YAML integer.")
    return value


def validate_config(config: dict, projections: int) -> None:
    volume, geometry = config["ReconVolume"], config["ScanGeometry"]
    for key in ("StartPosition", "EndPosition", "XOrigin", "YOrigin"):
        number(volume, key)
    for key in ("FieldOfView", "SliceThickness", "SlicePitch"):
        number(volume, key, positive=True)
    for key in ("Nx", "Ny"):
        value = number(volume, key, positive=True, integer=True)
        if not 16 <= value <= 4096:
            raise ValueError(f"{key} must be between 16 and 4096.")
    slices = abs(volume["EndPosition"] - volume["StartPosition"]) / volume["SlicePitch"] + 1
    if slices > 32767:
        raise ValueError("Requested volume exceeds FreeCT's NIfTI-1 slice-count limit.")
    for key in ("DetectorRows", "DetectorChannels", "ProjectionsPerRotation", "NumProjectionsTotal"):
        number(geometry, key, positive=True, integer=True)
    if geometry["DetectorRows"] < 10 or geometry["DetectorChannels"] < 2:
        raise ValueError("This FreeCT build requires >=10 detector rows and >=2 channels.")
    if geometry["NumProjectionsTotal"] != projections:
        raise ValueError("NumProjectionsTotal must equal the number of staged projections.")
    rotation = geometry["ProjectionsPerRotation"]
    if rotation < 4 or rotation % 2 or projections < rotation:
        raise ValueError("Require an even ProjectionsPerRotation >=4 and at least one full rotation.")
    for key in ("DetectorTransverseSpacing", "DetectorAxialSpacing", "DistSourceToDetector",
                "DistSourceToIsocenter", "AcquisitionFieldOfView", "CollimatedSliceWidth"):
        number(geometry, key, positive=True)
    if geometry["DistSourceToDetector"] <= geometry["DistSourceToIsocenter"]:
        raise ValueError("Source-to-detector distance must exceed source-to-isocenter distance.")
    for key in ("DetectorCentralRow", "DetectorCentralChannel", "StartTablePosition",
                "EndTablePosition", "TableFeedPerRotation", "HUCalibrationFactor"):
        number(geometry, key)
    if geometry.get("DetectorShape") != "CYLINDRICAL":
        raise ValueError("Only CYLINDRICAL detector geometry is supported by this adapter.")
    if geometry.get("ProjectionGeometry") != "FANBEAM":
        raise ValueError("Only FANBEAM projection geometry is supported by this adapter.")
    if geometry.get("FlyingFocalSpotMode") != "FFSNONE":
        raise ValueError("Flying focal spots are not supported by this adapter/build.")
    if geometry.get("ScanType") not in {"HELICAL", "AXIAL"}:
        raise ValueError("ScanType must be HELICAL or AXIAL for this build.")
    if geometry["ScanType"] == "HELICAL":
        if geometry["TableFeedPerRotation"] == 0:
            raise ValueError("Helical table feed cannot be zero.")
        travel = geometry["EndTablePosition"] - geometry["StartTablePosition"]
        if travel == 0 or (volume["EndPosition"] - volume["StartPosition"]) * travel < 0:
            raise ValueError("Reconstruction range must follow the helical acquisition direction.")
    specific = config["ReconSpecific"]
    a, c = number(specific, "KernelA"), number(specific, "KernelC")
    if a not in {0.5, 0.54, 1.0} or not 0 < c <= 1:
        raise ValueError("KernelA must be 0.5, 0.54 or 1.0; KernelC must be in (0, 1].")
    # Go's RestoreWFBPConfig type-asserts float64; YAML '1' is NOT sufficient.
    specific.update(KernelA=float(a), KernelC=float(c))
    output = config["ImageOutput"]
    if type(output.get("RescaleToHU")) is not bool:
        raise ValueError("RescaleToHU must be true or false.")
    if output["RescaleToHU"] and geometry["HUCalibrationFactor"] <= 0:
        raise ValueError("HU output requires a finite positive HUCalibrationFactor; do not guess it.")
    if output.get("DICOM") and not output["RescaleToHU"]:
        raise ValueError("FreeCT's native DICOM writer requires HU rescaling.")
    number(output, "AngleOffset")
    for key in ("CorrectMATLABIndexing", "FlipDetectorChannels", "FlipDetectorRows", "NegateTubeAngle"):
        if type(config["Preprocessing"].get(key)) is not bool:
            raise ValueError(f"Preprocessing.{key} must be true or false.")
    if number(config["Preprocessing"], "AdapativeFiltration") != 0:
        raise ValueError("Adaptive filtration is not implemented in the supplied FreeCT source.")


def bind_paths(config: dict, projections: Path, output: Path, native_dicom: bool) -> dict:
    """Always constrain engine I/O to this job, including imported/edited YAML."""
    config = copy.deepcopy(config)
    config["RawInput"] = {"Battelle": str(projections), "Binary": "", "Ptr": "", "Ctd": ""}
    config["ImageOutput"].update(
        Binary="", Png="", DICOM=str(output / "dicom_native") if native_dicom else "",
        NIfTI1=str(output / "freect_native.nii.gz"),
    )
    return config
