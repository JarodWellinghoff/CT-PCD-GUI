import os
import numpy as np
from pydicom import dcmread
from pydicom.tag import Tag
from collections import defaultdict


def load_ctpd_dictionary(dict_path):
    """Load custom DICOM-CT-PD dictionary"""
    tag_to_names = defaultdict(list)

    if not os.path.exists(dict_path):
        print(f"Warning: Dictionary file not found: {dict_path}")
        return tag_to_names

    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '(' not in line:
                continue
            try:
                parts = line.split()
                tag_str = parts[0]
                name = parts[2]
                g, e = tag_str.strip('()').split(',')
                tag = (int(g, 16), int(e, 16))
                tag_to_names[tag].append(name)
            except Exception:
                continue

    print(f"Loaded {len(tag_to_names)} named tags from dictionary.")
    return tag_to_names


def _decode_ctpd_value(value):
    """
    Convert raw private-tag values to readable Python types.
    Handles both string (LO/CS/DS) and binary (US/SS) representations.
    """
    if not isinstance(value, (bytes, bytearray)):
        return value

    # 1. Try string decoding first
    try:
        text = value.decode('ascii', errors='ignore').strip()
        if text and any(c.isprintable() and not c.isspace() for c in text):
            try:
                return int(text)
            except ValueError:
                pass
            try:
                return float(text)
            except ValueError:
                pass
            return text
    except Exception:
        pass

    # 2. Binary integer (US / SS / UL …)
    if len(value) == 2:
        return int.from_bytes(value, byteorder='little', signed=False)
    if len(value) == 4:
        return int.from_bytes(value, byteorder='little', signed=False)

    return value


def DICOMCTPDreader_PCD_MultiFrame(input_file, dictionary_path=None):
    """
    Reader for multi-frame PCD DICOM-CT-PD files.

    Returns
    -------
    projection3D : np.ndarray
        Shape (NumberOfFrames, Rows, Columns), after Rescale
    header : pydicom.Dataset
        Full DICOM header
    per_frame : dict
        Dictionary containing per-frame arrays:
            - DetectorFocalCenterAngularAll
            - DetectorFocalCenterAxialAll
            - DetectorFocalCenterRadialAll
            - SourceAngularPositionShiftAll
            - TimestampAll
            - PhotonStatisticsAll
            - InstanceNumberAll   ← projection order in the full scan
    """
    print("=== PCD MULTI-FRAME DICOM-CT-PD READER ===\n")

    tag_to_names = load_ctpd_dictionary(dictionary_path) if dictionary_path else {}

    ds = dcmread(input_file, force=True)

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    print(f"Patient Name         : {ds.get('PatientName', 'N/A')}")
    print(f"Modality             : {ds.get('Modality', 'N/A')}")
    print(f"Manufacturer         : {ds.get('Manufacturer', 'N/A')}")
    print(f"Rows x Columns       : {ds.get('Rows', 'N/A')} x {ds.get('Columns', 'N/A')}")
    print(f"Number of Frames     : {ds.get('NumberOfFrames', 'N/A')}")

    # ------------------------------------------------------------------
    # Shared (root-level) CTPD metadata
    # ------------------------------------------------------------------
    print("\n--- Important CTPD Metadata (root level) ---")

    important_tags = [
        (0x7033, 0x0001),   # DictionaryVersion
        (0x7033, 0x1059),   # TypeOfMECT
        (0x7033, 0x105B),   # NumberofSources
        (0x7033, 0x105D),   # SourceIndex
        (0x7033, 0x1061),   # NumberofSpectra
        (0x7033, 0x1063),   # SpectrumIndex
        (0x7033, 0x1062),   # ListOfSpectrum
        (0x7033, 0x1066),   # PhotonStatistics_mA
    ]

    for tag in important_tags:
        if tag in ds:
            elem = ds[tag]
            name_list = tag_to_names.get(tag, [f"Private_{tag[0]:04X}_{tag[1]:04X}"])
            display_name = name_list[-1]

            value = _decode_ctpd_value(elem.value)

            if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                print(f"{display_name:<35} : [{len(value)} elements]")
            else:
                print(f"{display_name:<35} : {value}")

    # ------------------------------------------------------------------
    # Per-frame private tags – extract ALL frames
    # ------------------------------------------------------------------
    print("\n--- Extracting per-frame private tags ---")

    n_frames = int(ds.NumberOfFrames)

    DetectorFocalCenterAngularAll = np.full(n_frames, np.nan, dtype=np.float32)
    DetectorFocalCenterAxialAll   = np.full(n_frames, np.nan, dtype=np.float32)
    DetectorFocalCenterRadialAll  = np.full(n_frames, np.nan, dtype=np.float32)
    SourceAngularPositionShiftAll = np.full(n_frames, np.nan, dtype=np.float32)
    TimestampAll                  = np.full(n_frames, np.nan, dtype=np.float32)
    InstanceNumberAll             = np.full(n_frames, -1, dtype=np.int32)  # -1 = missing
    PhotonStatisticsAll           = [None] * n_frames

    tags = {
        'DetectorFocalCenterAngularPosition': Tag(0x7031, 0x1001),
        'DetectorFocalCenterAxialPosition'  : Tag(0x7031, 0x1002),
        'DetectorFocalCenterRadialDistance' : Tag(0x7031, 0x1003),
        'SourceAngularPositionShift'        : Tag(0x7033, 0x100B),
        'PhotonStatistics'                  : Tag(0x7033, 0x1065),
        'Timestamp'                         : Tag(0x7033, 0x1067),
        'InstanceNumber'                    : Tag(0x0020, 0x0013),  # projection order
    }

    if hasattr(ds, 'PerFrameFunctionalGroupsSequence') and ds.PerFrameFunctionalGroupsSequence:
        for i, item in enumerate(ds.PerFrameFunctionalGroupsSequence):

            # Angular position
            t = tags['DetectorFocalCenterAngularPosition']
            if t in item:
                val = item[t].value
                DetectorFocalCenterAngularAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            # Axial position
            t = tags['DetectorFocalCenterAxialPosition']
            if t in item:
                val = item[t].value
                DetectorFocalCenterAxialAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            # Radial distance
            t = tags['DetectorFocalCenterRadialDistance']
            if t in item:
                val = item[t].value
                DetectorFocalCenterRadialAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            # Source angular shift
            t = tags['SourceAngularPositionShift']
            if t in item:
                val = item[t].value
                SourceAngularPositionShiftAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            # Timestamp
            t = tags['Timestamp']
            if t in item:
                val = item[t].value
                TimestampAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            # InstanceNumber (projection order in the full scan)
            t = tags['InstanceNumber']
            if t in item:
                val = item[t].value
                try:
                    InstanceNumberAll[i] = int(
                        val[0] if isinstance(val, (list, np.ndarray)) else val
                    )
                except Exception:
                    InstanceNumberAll[i] = -1

            # PhotonStatistics
            t = tags['PhotonStatistics']
            if t in item:
                val = item[t].value
                if isinstance(val, (bytes, bytearray)):
                    PhotonStatisticsAll[i] = np.frombuffer(val, dtype=np.float32).copy()
                else:
                    PhotonStatisticsAll[i] = np.asarray(val, dtype=np.float32).ravel()

        print(f"Successfully extracted data from all {n_frames} frames.")
        print(f"  DetectorFocalCenterAngularPosition[0] = {DetectorFocalCenterAngularAll[0]}")
        print(f"  DetectorFocalCenterAxialPosition[0]   = {DetectorFocalCenterAxialAll[0]}")
        print(f"  DetectorFocalCenterRadialDistance[0]  = {DetectorFocalCenterRadialAll[0]}")
        print(f"  Timestamp[0]                          = {TimestampAll[0]}")
        print(f"  InstanceNumber[0]                     = {InstanceNumberAll[0]}")
        if n_frames > 1:
            print(f"  InstanceNumber range                  = "
                  f"{InstanceNumberAll[InstanceNumberAll >= 0].min() if np.any(InstanceNumberAll >= 0) else 'N/A'}"
                  f" … "
                  f"{InstanceNumberAll[InstanceNumberAll >= 0].max() if np.any(InstanceNumberAll >= 0) else 'N/A'}")
        if PhotonStatisticsAll[0] is not None:
            print(f"  PhotonStatistics[0] length            = {len(PhotonStatisticsAll[0])}")
    else:
        print("  PerFrameFunctionalGroupsSequence not present")

    # ------------------------------------------------------------------
    # Projection data
    # ------------------------------------------------------------------
    print("\nReading projection data...")
    raw_data = ds.pixel_array

    if raw_data.ndim == 2:
        raw_data = raw_data[np.newaxis, ...]
    elif raw_data.ndim == 3 and raw_data.shape[2] == n_frames:
        raw_data = np.moveaxis(raw_data, 2, 0)

    projection3D = raw_data.astype(np.float64)

    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        projection3D = projection3D * slope + intercept
        print(f"Rescale applied      : Slope = {slope:.8f}, Intercept = {intercept:.8f}")
    else:
        print("Warning: Rescale tags not found!")

    print(f"Final Projection Shape : {projection3D.shape}")
    print("\nReader finished successfully.")

    # ------------------------------------------------------------------
    # Pack per-frame results
    # ------------------------------------------------------------------
    per_frame = {
        'DetectorFocalCenterAngularAll': DetectorFocalCenterAngularAll,
        'DetectorFocalCenterAxialAll'  : DetectorFocalCenterAxialAll,
        'DetectorFocalCenterRadialAll' : DetectorFocalCenterRadialAll,
        'SourceAngularPositionShiftAll': SourceAngularPositionShiftAll,
        'TimestampAll'                 : TimestampAll,
        'InstanceNumberAll'            : InstanceNumberAll,  # projection order
        'PhotonStatisticsAll'          : PhotonStatisticsAll,
    }

    return projection3D, ds, per_frame


# ====================== Example Usage ======================
if __name__ == "__main__":
    input_file = r'F:\PCD_DICOMCTPD_Generator\Batch_00001_multiframe_ct_pd.dcm'
    dict_path  = r'F:\PCD_DICOMCTPD_Generator\DICOM-CT-PD-dict_v11.txt'

    projection3D, header, per_frame = DICOMCTPDreader_PCD_MultiFrame(input_file, dict_path)

    # Example access
    print("\n=== Quick access example ===")
    print(f"Angular position of frame 0 : {per_frame['DetectorFocalCenterAngularAll'][0]}")
    print(f"InstanceNumber of frame 0   : {per_frame['InstanceNumberAll'][0]}")
    print(f"PhotonStatistics of frame 0 length : {len(per_frame['PhotonStatisticsAll'][0])}")

    # Projection data
    print(projection3D.shape)  # (2000, 2752, 120)

    # Per-frame geometry
    angles   = per_frame['DetectorFocalCenterAngularAll']
    axial    = per_frame['DetectorFocalCenterAxialAll']
    radial   = per_frame['DetectorFocalCenterRadialAll']
    shifts   = per_frame['SourceAngularPositionShiftAll']
    times    = per_frame['TimestampAll']
    inst_num = per_frame['InstanceNumberAll']   # e.g. 1…2000 or 2001…4000
    photon   = per_frame['PhotonStatisticsAll']  # list of arrays