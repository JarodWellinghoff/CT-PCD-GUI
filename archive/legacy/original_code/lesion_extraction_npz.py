import os
import numpy as np
import nrrd
import pydicom
import json
import glob
import SimpleITK as sitk
import pandas as pd
from openpyxl.styles import Font
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Most targeted — suppress only this specific message:
warnings.filterwarnings("ignore", message="Invalid value for VR")

WORKERS = None
# DICOM header fields to pull into the spreadsheet (skip silently if missing).
DICOM_HEADER_FIELDS = (
    "PatientID",
    "PatientName",
    "StudyDate",
    "SeriesDescription",
    "Manufacturer",
    "ManufacturerModelName",
    "KVP",
    "XRayTubeCurrent",
    "Exposure",
    "SliceThickness",
    "PixelSpacing",
    "ConvolutionKernel",
)


def _process_one(args):
    """Top-level wrapper so the pool can pickle it. Returns (rows, error_str)."""
    nrrd_path, dcm_path, npz_path = args
    base_info = {"nrrd_path": nrrd_path, "dcm_path": dcm_path, "npz_path": npz_path}
    if not os.path.exists(nrrd_path):
        return [{**base_info, "SkipReason": "Missing NRRD"}], None
    if not os.path.exists(dcm_path):
        return [{**base_info, "SkipReason": "Missing DICOM"}], None
    try:
        return main(nrrd_path, dcm_path, npz_path), None
    except Exception as e:
        return [{**base_info, "SkipReason": f"Exception: {e}"}], str(e)


def safe_get(func, *args, default=None):
    """Call func(*args); return default on any exception."""
    try:
        return func(*args)
    except Exception:
        return default


def _optional_numeric_array(value):
    """Convert an optional numeric value to an NPZ-safe array without pickling."""
    if value is None:
        return np.asarray(np.nan, dtype=np.float64)
    return np.asarray(value)


def get_dicom_header(dcm_path, dicom_name):
    DicomHeader = dict()
    temp_dcm = pydicom.dcmread(os.path.join(dcm_path, dicom_name), force=True)
    for element in temp_dcm:
        if (
            ("Private" not in element.name)
            and ("Pixel Data" != element.name)
            and ("CSA Image" not in element.name)
        ):
            if len(element.name) > 31:
                temp_field = element.name[0:31]
            else:
                temp_field = element.name
            temp_field = str.join("", str.split(temp_field))
            if "DSfloat" in str(type(element.value)):
                DicomHeader[temp_field] = float(element.value)
            elif "MultiValue" in str(type(element.value)):
                temp_value = list(element.value)
                for temp_value_idx in np.arange(0, len(temp_value)):
                    if "DSfloat" in str(type(temp_value[temp_value_idx])):
                        temp_value[temp_value_idx] = float(temp_value[temp_value_idx])
                    else:
                        temp_value[temp_value_idx] = str(temp_value[temp_value_idx])
                DicomHeader[temp_field] = temp_value
            else:
                DicomHeader[temp_field] = str(element.value)
    return DicomHeader


def _header_subset(DicomHeader):
    """Pull a subset of useful DICOM header fields, stringifying MultiValue entries."""
    out = {}
    for k in DICOM_HEADER_FIELDS:
        if k in DicomHeader:
            v = DicomHeader[k]
            if isinstance(v, list):
                v = ";".join(str(x) for x in v)
            out[f"DICOM_{k}"] = v
    return out


def main(nrrd_path, dcm_path, npz_path):
    """
    Process one case. Returns a list of row dicts (one per lesion, or a single
    blank row if the case is skipped).
    """
    os.makedirs(npz_path, exist_ok=True)
    rows = []
    base_info = {
        "nrrd_path": nrrd_path,
        "dcm_path": dcm_path,
        "npz_path": npz_path,
    }

    # Load DICOM series
    dicom_reader = sitk.ImageSeriesReader()
    dicom_names = dicom_reader.GetGDCMSeriesFileNames(dcm_path)
    dicom_reader.SetFileNames(dicom_names)
    dicom_image = dicom_reader.Execute()

    # Load NRRD segmentation
    nrrd_reader = sitk.ImageFileReader()
    nrrd_reader.SetImageIO("NrrdImageIO")
    nrrd_reader.SetFileName(nrrd_path)
    nrrd_image = nrrd_reader.Execute()

    if nrrd_image.GetNumberOfComponentsPerPixel() > 1:
        # print(
        #     f"Warning: NRRD image has {nrrd_image.GetNumberOfComponentsPerPixel()} "
        #     f"components per pixel, expected 1. Resegmenting manually is recommended."
        # )
        # print("Skipping this case.")
        rows.append({**base_info, "SkipReason": "Multi-component NRRD"})
        return rows

    dicom_image = sitk.Cast(dicom_image, sitk.sitkInt16)
    nrrd_image = sitk.Cast(nrrd_image, sitk.sitkUInt8)

    _, header = nrrd.read(nrrd_path)

    # First pass: collect lesion label info from the NRRD header
    lesions = []
    for key, item in header.items():
        if "name" in key.lower() and "-lesion" in item.lower():
            lesion_label = key.split("_")[0]
            lesion_int = int(header[f"{lesion_label}_LabelValue"])
            lesions.append({"label": lesion_label, "int": lesion_int, "name": item})

    if not lesions:
        rows.append({**base_info, "SkipReason": "No lesion labels in NRRD header"})
        return rows

    lesions = sorted(lesions, key=lambda x: x["name"])

    # Build a mask containing only lesion labels (others -> 0)
    lesion_ints = [l["int"] for l in lesions]
    arr = sitk.GetArrayViewFromImage(nrrd_image)
    arr_filtered = np.where(np.isin(arr, lesion_ints), arr, 0).astype(arr.dtype)
    lesion_only = sitk.GetImageFromArray(arr_filtered)
    lesion_only.CopyInformation(nrrd_image)

    lesion_label_stats_filter = sitk.LabelStatisticsImageFilter()
    lesion_label_stats_filter.Execute(dicom_image, lesion_only)

    lesion_label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    lesion_label_shape_filter.Execute(lesion_only)

    DicomHeader = get_dicom_header(dcm_path, dicom_names[0])
    header_subset = _header_subset(DicomHeader)

    # zyx -> xyz to preserve the original export orientation
    lesion_only_array = sitk.GetArrayViewFromImage(lesion_only).transpose(1, 2, 0)
    dicom_image_array = sitk.GetArrayViewFromImage(dicom_image).transpose(1, 2, 0)

    for temp_lesion_idx, lesion in enumerate(lesions):
        row = {
            **base_info,
            **header_subset,
            "LesionName": lesion["name"],
            "LesionLabel": lesion["label"],
            "LesionLabelValue": lesion["int"],
            "LesionNumber": temp_lesion_idx,
        }

        # Bounding box (in xyz order from SimpleITK: x_min,x_max,y_min,y_max,z_min,z_max)
        lesion_bbox_raw = safe_get(
            lesion_label_stats_filter.GetBoundingBox, lesion["int"]
        )
        lesion_bbox = None
        if lesion_bbox_raw is not None:
            lesion_bbox = np.array(lesion_bbox_raw).reshape(3, 2)
            row["BoundingBox_x_min"] = int(lesion_bbox[0, 0])
            row["BoundingBox_x_max"] = int(lesion_bbox[0, 1])
            row["BoundingBox_y_min"] = int(lesion_bbox[1, 0])
            row["BoundingBox_y_max"] = int(lesion_bbox[1, 1])
            row["BoundingBox_z_min"] = int(lesion_bbox[2, 0])
            row["BoundingBox_z_max"] = int(lesion_bbox[2, 1])

        # 2D-safe statistics (work even for single-slice lesions)
        mean_hu = safe_get(lesion_label_stats_filter.GetMean, lesion["int"])
        max_hu = safe_get(lesion_label_stats_filter.GetMaximum, lesion["int"])
        min_hu = safe_get(lesion_label_stats_filter.GetMinimum, lesion["int"])
        median_hu = safe_get(lesion_label_stats_filter.GetMedian, lesion["int"])
        sigma = safe_get(lesion_label_stats_filter.GetSigma, lesion["int"])
        variance = safe_get(lesion_label_stats_filter.GetVariance, lesion["int"])
        voxel_count = safe_get(lesion_label_stats_filter.GetCount, lesion["int"])
        physical_size = safe_get(
            lesion_label_shape_filter.GetPhysicalSize, lesion["int"]
        )

        row.update(
            {
                "LesionMeanHU": mean_hu,
                "LesionMaxHU": max_hu,
                "LesionMinHU": min_hu,
                "LesionMedianHU": median_hu,
                "LesionSigma": sigma,
                "LesionVariance": variance,
                "LesionVoxelCount": voxel_count,
                "LesionPhysicalSize": physical_size,
            }
        )

        # Detect single-slice lesion -> emit a row with what we have, skip .npz
        single_slice = False
        region = safe_get(lesion_label_stats_filter.GetRegion, lesion["int"])
        if region is not None:
            single_slice = (np.array(region[3:]) == 1).any()

        if single_slice:
            # print(
            #     f"Warning: Lesion {lesion['name']} has only 1 slice; "
            #     "skipping 3D features and .npz export."
            # )
            row["SkipReason"] = "Single-slice lesion (3D features skipped)"
            rows.append(row)
            continue

        # 3D features
        ellipsoid_diameters = safe_get(
            lesion_label_shape_filter.GetEquivalentEllipsoidDiameter, lesion["int"]
        )
        if ellipsoid_diameters is not None:
            row["LesionDiameter_0"] = ellipsoid_diameters[0]
            row["LesionDiameter_1"] = ellipsoid_diameters[1]
            row["LesionDiameter_2"] = ellipsoid_diameters[2]

        roundness = safe_get(lesion_label_shape_filter.GetRoundness, lesion["int"])
        row["LesionRoundness"] = roundness

        centroid_phys = safe_get(lesion_label_shape_filter.GetCentroid, lesion["int"])
        center = None
        if centroid_phys is not None:
            center = dicom_image.TransformPhysicalPointToIndex(centroid_phys)
            row["LesionCenter_x"] = int(center[0])
            row["LesionCenter_y"] = int(center[1])
            row["LesionCenter_z"] = int(center[2])

        # print(
        #     f"Lesion: {lesion['name']}, Label: {lesion['label']}, Bounding Box: {bbox}"
        # )

        # Save .npz (preserves original behavior)
        if lesion_bbox is not None and center is not None:
            padding = np.ceil(np.diff(lesion_bbox, axis=1)[:, 0] / 2).astype(int)
            padding = np.array([-padding, padding]).T

            lesion_bbox_padded = lesion_bbox + padding
            binary_lesion_mask = lesion_only_array[
                lesion_bbox_padded[1, 0] : lesion_bbox_padded[1, 1] + 1,
                lesion_bbox_padded[0, 0] : lesion_bbox_padded[0, 1] + 1,
                lesion_bbox_padded[2, 0] : lesion_bbox_padded[2, 1] + 1,
            ]
            binary_lesion_mask = (binary_lesion_mask == lesion["int"]).astype(bool)
            voi = dicom_image_array[
                lesion_bbox_padded[1, 0] : lesion_bbox_padded[1, 1] + 1,
                lesion_bbox_padded[0, 0] : lesion_bbox_padded[0, 1] + 1,
                lesion_bbox_padded[2, 0] : lesion_bbox_padded[2, 1] + 1,
            ]
            # voi = np.multiply(voi, binary_lesion_mask)

            # NPZ is a flat collection of named arrays. The original nested
            # MATLAB structure is flattened, and the DICOM header is stored as
            # JSON so the file can be loaded with allow_pickle=False.
            lesion_diameter = (
                np.asarray(ellipsoid_diameters, dtype=np.float64)
                if ellipsoid_diameters is not None
                else np.empty(0, dtype=np.float64)
            )
            temp_det_file = os.path.join(npz_path, f"{lesion['name']}.npz")
            np.savez_compressed(
                temp_det_file,
                PatientName=np.asarray(lesion["name"]),
                LesionNumber=np.asarray(temp_lesion_idx, dtype=np.int64),
                Org_case_path=np.asarray(dcm_path),
                Org_mask=np.asarray(nrrd_path),
                Org_dcm_path=np.asarray(dcm_path),
                DicomHeaderJSON=np.asarray(json.dumps(DicomHeader, ensure_ascii=False)),
                LesionMask=binary_lesion_mask,
                VOI=voi,
                org_slice_rng=np.asarray(lesion_bbox[2, :], dtype=np.int64),
                org_col_rng=np.asarray(lesion_bbox[0, :], dtype=np.int64),
                LesionCenter=np.asarray(center, dtype=np.int64),
                org_row_rng=np.asarray(lesion_bbox[1, :], dtype=np.int64),
                LesionDiameter=lesion_diameter,
                LesionMeanHU=_optional_numeric_array(mean_hu),
                LesionRoundness=_optional_numeric_array(roundness),
                LesionMaxHU=_optional_numeric_array(max_hu),
                LesionMinHU=_optional_numeric_array(min_hu),
                LesionMedianHU=_optional_numeric_array(median_hu),
                LesionSigma=_optional_numeric_array(sigma),
                LesionVariance=_optional_numeric_array(variance),
                LesionVoxelCount=_optional_numeric_array(voxel_count),
                LesionPhysicalSize=_optional_numeric_array(physical_size),
            )

        rows.append(row)

    # print("==" * 50)
    return rows


def write_excel(rows, excel_output):
    """Write rows to xlsx with column ordering, bold header, frozen pane, autosized columns."""
    # Preferred column order; any extra keys are appended at the end.
    preferred_order = [
        "nrrd_path",
        "dcm_path",
        "npz_path",
        "LesionName",
        "LesionLabel",
        "LesionLabelValue",
        "LesionNumber",
        "DICOM_PatientID",
        "DICOM_PatientName",
        "DICOM_StudyDate",
        "DICOM_SeriesDescription",
        "DICOM_Manufacturer",
        "DICOM_ManufacturerModelName",
        "DICOM_KVP",
        "DICOM_XRayTubeCurrent",
        "DICOM_Exposure",
        "DICOM_SliceThickness",
        "DICOM_PixelSpacing",
        "DICOM_ConvolutionKernel",
        "BoundingBox_x_min",
        "BoundingBox_x_max",
        "BoundingBox_y_min",
        "BoundingBox_y_max",
        "BoundingBox_z_min",
        "BoundingBox_z_max",
        "LesionCenter_x",
        "LesionCenter_y",
        "LesionCenter_z",
        "LesionDiameter_0",
        "LesionDiameter_1",
        "LesionDiameter_2",
        "LesionMeanHU",
        "LesionMaxHU",
        "LesionMinHU",
        "LesionMedianHU",
        "LesionSigma",
        "LesionVariance",
        "LesionVoxelCount",
        "LesionPhysicalSize",
        "LesionRoundness",
        "SkipReason",
    ]
    df = pd.DataFrame(rows)
    cols = [c for c in preferred_order if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lesions")
        ws = writer.sheets["Lesions"]
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            values = [c.value for c in col_cells if c.value is not None]
            max_len = max((len(str(v)) for v in values), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(
                max(max_len + 2, 10), 50
            )
    print(f"Wrote {len(df)} rows to {excel_output}")


if __name__ == "__main__":
    parent_dir = r"U:\PUBLIC\Lesion_library\Liver\Liver_P2S2_Scott_40cases"
    nrrd_dir = os.path.join(parent_dir, "Segmentations", "**", "MANUAL", "*.nrrd")
    excel_output = os.path.join(
        parent_dir, "Segmentations", "Documents", "lesion_summary.xlsx"
    )
    nrrd_paths = sorted(glob.glob(nrrd_dir))
    dcm_paths = [
        p.split("\\MANUAL\\")[0].replace("Segmentations", "DICOMS") for p in nrrd_paths
    ]
    npz_paths = [p.replace("DICOMS", "Lesions_npz") for p in dcm_paths]
    # nrrd_paths = [nrrd_paths[0]]
    # dcm_paths = [dcm_paths[0]]
    # npz_paths = [npz_paths[0]]
    tasks = list(zip(nrrd_paths, dcm_paths, npz_paths))
    all_rows = []
    if WORKERS == 1:
        for t in tqdm(tasks, desc="Cases", unit="case"):
            rows, err = _process_one(t)
            all_rows.extend(rows)
            if err:
                tqdm.write(f"ERROR {os.path.basename(t[0])}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(_process_one, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Cases", unit="case") as pbar:
                for fut in as_completed(futures):
                    nrrd_path, _, _ = futures[fut]
                    rows, err = fut.result()
                    all_rows.extend(rows)
                    if err:
                        tqdm.write(f"ERROR {os.path.basename(nrrd_path)}: {err}")
                    pbar.set_postfix_str(os.path.basename(nrrd_path)[:40])
                    pbar.update(1)

    write_excel(all_rows, excel_output)
