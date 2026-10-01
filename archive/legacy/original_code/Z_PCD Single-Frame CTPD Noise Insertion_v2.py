"""
PCD DICOM-CT-PD multi-frame projection-domain noise insertion

Workflow
--------
1. Read multi-frame CTPD (attenuation + per-frame PhotonStatistics)
2. For each frame (in-place, float32 volume):
       P_B = P_A + sqrt((1/N2 - 1/N1)*(1 + Ne/N2 + Ne/N1)) * randn
   with N1_det = N1_incident * exp(-P_A),   N2_det = N1_det * mAsFactor * fineTuneFactor

   mAsFactor is the nominal dose fraction (e.g. 0.25 = 25% dose).
   Because PhotonStatistics is an NEPN estimate, not exact counts, noise
   after using mAsFactor alone may not match a true 25% acquisition.
   fineTuneFactor is an extra calibration gain so users can match
   measured ROI SD without changing the nominal dose label.
3. Encode with ORIGINAL RescaleIntercept / RescaleSlope (same as FD)

Header layout matches FD as closely as possible:
- RescaleIntercept / RescaleSlope kept from input
- Root PixelRepresentation / Smallest / Largest only if present in FD
- Per-frame Smallest/Largest updated from noisy pixels
- Per-frame InstanceNumber preserved (projection order in full scan)
- SeriesDescription updated FD -> SIM xxpct
"""

"""
Zhongxing Zhou modified
September 2026
    Dose calibration (fineTuneFactor)
    •	Detected counts:
        N2_det = N1_det * mAsFactor * fineTuneFactor
    •	mAsFactor is the nominal dose fraction (e.g. 0.25 = 25% dose).   Because PhotonStatistics is an NEPN estimate.
        After using mAsFactor alone may not match a true 25% acquisition.    
        fineTuneFactor is an extra calibration gain so users can match measured ROI SD without changing the nominal dose label.

"""

import os
import numpy as np
from copy import deepcopy
from collections import defaultdict

from pydicom import dcmread, dcmwrite, Dataset
from pydicom.tag import Tag
from pydicom.uid import generate_uid, ExplicitVRLittleEndian


# ======================================================================
# Multi-frame reader
# ======================================================================
def load_ctpd_dictionary(dict_path):
    tag_to_names = defaultdict(list)
    if not os.path.exists(dict_path):
        print(f"Warning: Dictionary file not found: {dict_path}")
        return tag_to_names
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "(" not in line:
                continue
            try:
                parts = line.split()
                tag_str = parts[0]
                name = parts[2]
                g, e = tag_str.strip("()").split(",")
                tag = (int(g, 16), int(e, 16))
                tag_to_names[tag].append(name)
            except Exception:
                continue
    print(f"Loaded {len(tag_to_names)} named tags from dictionary.")
    return tag_to_names


def _decode_ctpd_value(value):
    if not isinstance(value, (bytes, bytearray)):
        return value
    try:
        text = value.decode("ascii", errors="ignore").strip()
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
    if len(value) == 2:
        return int.from_bytes(value, byteorder="little", signed=False)
    if len(value) == 4:
        return int.from_bytes(value, byteorder="little", signed=False)
    return value


def DICOMCTPDreader_PCD_MultiFrame(input_file, dictionary_path=None):
    print("=== PCD MULTI-FRAME DICOM-CT-PD READER ===\n")
    tag_to_names = load_ctpd_dictionary(dictionary_path) if dictionary_path else {}
    ds = dcmread(input_file, force=True)

    print(f"Patient Name         : {ds.get('PatientName', 'N/A')}")
    print(f"Modality             : {ds.get('Modality', 'N/A')}")
    print(f"Manufacturer         : {ds.get('Manufacturer', 'N/A')}")
    print(f"Series Description   : {ds.get('SeriesDescription', 'N/A')}")
    print(
        f"Rows x Columns       : {ds.get('Rows', 'N/A')} x {ds.get('Columns', 'N/A')}"
    )
    print(f"Number of Frames     : {ds.get('NumberOfFrames', 'N/A')}")

    print("\n--- Important CTPD Metadata (root level) ---")
    important_tags = [
        (0x7033, 0x0001),
        (0x7033, 0x1059),
        (0x7033, 0x105B),
        (0x7033, 0x105D),
        (0x7033, 0x1061),
        (0x7033, 0x1063),
        (0x7033, 0x1062),
        (0x7033, 0x1066),
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

    print("\n--- Extracting per-frame private tags ---")
    n_frames = int(ds.NumberOfFrames)

    DetectorFocalCenterAngularAll = np.full(n_frames, np.nan, dtype=np.float32)
    DetectorFocalCenterAxialAll = np.full(n_frames, np.nan, dtype=np.float32)
    DetectorFocalCenterRadialAll = np.full(n_frames, np.nan, dtype=np.float32)
    SourceAngularPositionShiftAll = np.full(n_frames, np.nan, dtype=np.float32)
    TimestampAll = np.full(n_frames, np.nan, dtype=np.float32)
    InstanceNumberAll = np.full(n_frames, -1, dtype=np.int32)  # -1 = missing
    PhotonStatisticsAll = [None] * n_frames

    tags = {
        "DetectorFocalCenterAngularPosition": Tag(0x7031, 0x1001),
        "DetectorFocalCenterAxialPosition": Tag(0x7031, 0x1002),
        "DetectorFocalCenterRadialDistance": Tag(0x7031, 0x1003),
        "SourceAngularPositionShift": Tag(0x7033, 0x100B),
        "PhotonStatistics": Tag(0x7033, 0x1065),
        "Timestamp": Tag(0x7033, 0x1067),
        "InstanceNumber": Tag(0x0020, 0x0013),  # projection order in full scan
    }

    if (
        hasattr(ds, "PerFrameFunctionalGroupsSequence")
        and ds.PerFrameFunctionalGroupsSequence
    ):
        for i, item in enumerate(ds.PerFrameFunctionalGroupsSequence):
            t = tags["DetectorFocalCenterAngularPosition"]
            if t in item:
                val = item[t].value
                DetectorFocalCenterAngularAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            t = tags["DetectorFocalCenterAxialPosition"]
            if t in item:
                val = item[t].value
                DetectorFocalCenterAxialAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            t = tags["DetectorFocalCenterRadialDistance"]
            if t in item:
                val = item[t].value
                DetectorFocalCenterRadialAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            t = tags["SourceAngularPositionShift"]
            if t in item:
                val = item[t].value
                SourceAngularPositionShiftAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            t = tags["Timestamp"]
            if t in item:
                val = item[t].value
                TimestampAll[i] = float(
                    val[0] if isinstance(val, (list, np.ndarray)) else val
                )

            # InstanceNumber (projection order)
            t = tags["InstanceNumber"]
            if t in item:
                val = item[t].value
                try:
                    InstanceNumberAll[i] = int(
                        val[0] if isinstance(val, (list, np.ndarray)) else val
                    )
                except Exception:
                    InstanceNumberAll[i] = -1

            t = tags["PhotonStatistics"]
            if t in item:
                val = item[t].value
                if isinstance(val, (bytes, bytearray)):
                    PhotonStatisticsAll[i] = np.frombuffer(val, dtype=np.float32).copy()
                else:
                    PhotonStatisticsAll[i] = np.asarray(val, dtype=np.float32).ravel()

        print(f"Successfully extracted data from all {n_frames} frames.")
        if PhotonStatisticsAll[0] is not None:
            print(f"  PhotonStatistics[0] length = {len(PhotonStatisticsAll[0])}")
        if np.any(InstanceNumberAll >= 0):
            valid = InstanceNumberAll[InstanceNumberAll >= 0]
            print(f"  InstanceNumber range       = {valid.min()} … {valid.max()}")
    else:
        print("  PerFrameFunctionalGroupsSequence not present")

    print("\nReading projection data...")
    raw_data = ds.pixel_array
    if raw_data.ndim == 2:
        raw_data = raw_data[np.newaxis, ...]
    elif raw_data.ndim == 3 and raw_data.shape[2] == n_frames:
        raw_data = np.moveaxis(raw_data, 2, 0)

    # float32 is sufficient for log-attenuation projections
    projection3D = raw_data.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        projection3D = projection3D * np.float32(slope) + np.float32(intercept)
        print(
            f"Rescale applied      : Slope = {slope:.8f}, Intercept = {intercept:.8f}"
        )
    else:
        print("Warning: Rescale tags not found!")

    print(f"Final Projection Shape : {projection3D.shape}")
    print("Reader finished successfully.\n")

    per_frame = {
        "DetectorFocalCenterAngularAll": DetectorFocalCenterAngularAll,
        "DetectorFocalCenterAxialAll": DetectorFocalCenterAxialAll,
        "DetectorFocalCenterRadialAll": DetectorFocalCenterRadialAll,
        "SourceAngularPositionShiftAll": SourceAngularPositionShiftAll,
        "TimestampAll": TimestampAll,
        "InstanceNumberAll": InstanceNumberAll,
        "PhotonStatisticsAll": PhotonStatisticsAll,
    }
    return projection3D, ds, per_frame


# ======================================================================
# Noise model (matches MATLAB)
# ======================================================================
def add_poisson_noise_log_domain(
    P_A, N1_incident, mAsFactor, fineTuneFactor=1.0, Ne=0.0, rng=None
):
    """
    P_B = P_A + sqrt((1/N2-1/N1)*(1+Ne/N2+Ne/N1)) * randn
    N1_det = N1_inc * exp(-P_A)
    N2_det = N1_det * mAsFactor * fineTuneFactor
    mAsFactor       : nominal dose fraction used in SeriesDescription (e.g. 0.25).
    fineTuneFactor  : calibration gain on N2. Default 1.0 leaves the model unchanged.
                      Use this when imperfect PhotonStatistics calibration makes
                      mAsFactor=0.25 produce a slightly wrong noise level.
    Internals use float64; return float32 for the stored volume.
    """
    if rng is None:
        rng = np.random.default_rng()
    if mAsFactor <= 0 or mAsFactor > 1:
        raise ValueError("mAsFactor must be in (0, 1].")
    if fineTuneFactor <= 0:
        raise ValueError("fineTuneFactor must be > 0.")

    P_A = np.asarray(P_A, dtype=np.float64)
    N1 = np.asarray(N1_incident, dtype=np.float64).ravel()
    H, W = P_A.shape

    if N1.size == H:
        N1_map = np.broadcast_to(N1[:, np.newaxis], (H, W))
    elif N1.size == W:
        N1_map = np.broadcast_to(N1[np.newaxis, :], (H, W))
    elif N1.size == H * W:
        N1_map = N1.reshape(H, W)
    else:
        raise ValueError(
            f"PhotonStatistics length ({N1.size}) does not match "
            f"frame shape {P_A.shape} on either axis."
        )

    atten = np.exp(-np.clip(P_A, -50, 50))
    N1_det = np.maximum(N1_map * atten, 1e-6)
    N2_det = np.maximum(N1_det * mAsFactor * fineTuneFactor, 1e-6)

    var_term = (1.0 / N2_det - 1.0 / N1_det) * (1.0 + Ne / N2_det + Ne / N1_det)
    var_term = np.maximum(var_term, 0.0)
    P_B = P_A + np.sqrt(var_term) * rng.standard_normal(size=P_A.shape)
    return P_B.astype(np.float32)


# ======================================================================
# Multi-frame noise insertion
# ======================================================================
TAG_SMALLEST = Tag(0x0028, 0x0106)
TAG_LARGEST = Tag(0x0028, 0x0107)
TAG_TUBE_CURRENT = Tag(0x0018, 0x1151)
TAG_INSTANCE_NUMBER = Tag(0x0020, 0x0013)


def _force_minmax_us(dataset):
    """Force Smallest/Largest Image Pixel Value to VR=US everywhere (root + nested)."""
    for elem in dataset.iterall():
        if elem.tag in (TAG_SMALLEST, TAG_LARGEST):
            elem.VR = "US"


def _encode_with_fixed_rescale(P_B, intercept, slope):
    """
    Encode attenuation using the original RescaleIntercept / RescaleSlope.
    Values outside the original dynamic range are clipped to [0, 65535].

    Returns
    -------
    pixel_u16 : ndarray uint16
    n_clipped : int
        Number of pixels that were clipped.
    """
    if slope == 0:
        raise ValueError("RescaleSlope is zero; cannot encode.")

    raw = (P_B - intercept) / slope
    n_clipped = int(np.sum((raw < 0) | (raw > 65535)))
    pixel_u16 = np.clip(np.round(raw), 0, 65535).astype(np.uint16)
    return pixel_u16, n_clipped


def insert_noise_multiframe(
    input_file,
    output_file,
    mAsFactor,
    dictionary_path=None,
    Ne=0.0,
    seed=None,
    update_tube_current=True,
    series_description=None,
    fineTuneFactor=1.0,
):
    """
    Insert noise into a multi-frame PCD DICOM-CT-PD file and write a new multi-frame file.

    RescaleIntercept / RescaleSlope are kept from the input (FD) file.
    Per-frame InstanceNumber is preserved (via deepcopy of PerFrameFunctionalGroupsSequence).
    SeriesDescription is updated from FD to SIM xxpct unless already marked SIM. (xxpct--xx percent of routine dose)
    fineTuneFactor : extra multiplier on N2_det = N1_det * mAsFactor * fineTuneFactor.
                     Does not change SeriesDescription (still "SIM 25pct" if mAsFactor=0.25).
                     Lets the user calibrate simulated noise to measured low-dose data.
    """
    rng = np.random.default_rng(seed)

    # ---- 1. Read ----
    projection3D, ds, per_frame = DICOMCTPDreader_PCD_MultiFrame(
        input_file, dictionary_path=dictionary_path
    )
    n_frames, H, W = projection3D.shape
    photon_all = per_frame["PhotonStatisticsAll"]

    # Original Rescale (must keep)
    if not (hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept")):
        raise KeyError(
            "RescaleSlope / RescaleIntercept not found in input multi-frame CTPD."
        )
    intercept = float(ds.RescaleIntercept)
    slope = float(ds.RescaleSlope)
    print(f"Keeping original Rescale: Intercept = {intercept:.8g}, Slope = {slope:.8g}")

    # Whether root min/max existed in FD
    had_root_smallest = TAG_SMALLEST in ds
    had_root_largest = TAG_LARGEST in ds

    # ---- 2. Noise per frame (in-place; no second volume) ----
    print(
        f"Inserting noise with mAsFactor = {mAsFactor}, fineTuneFactor = {fineTuneFactor} ..."
    )
    for i in range(n_frames):
        N1 = photon_all[i]
        if N1 is None:
            raise RuntimeError(
                f"Frame {i}: PhotonStatistics missing. "
                "Cannot perform NEPN-based noise insertion."
            )
        projection3D[i] = add_poisson_noise_log_domain(
            projection3D[i],
            N1,
            mAsFactor,
            fineTuneFactor=fineTuneFactor,
            Ne=Ne,
            rng=rng,
        )
        if (i + 1) % 200 == 0 or (i + 1) == n_frames:
            print(f"  Noised frame {i + 1}/{n_frames}")

    # ---- 3. Encode with FIXED (original) Rescale ----
    pixel_u16, n_clipped = _encode_with_fixed_rescale(projection3D, intercept, slope)
    if n_clipped > 0:
        print(
            f"Note: {n_clipped} pixel(s) were clipped to [0, 65535] "
            f"when encoding with the original Rescale range."
        )
    else:
        print("No pixels were clipped when encoding with the original Rescale range.")

    # ---- 4. Build output multi-frame dataset ----
    # deepcopy keeps PerFrameFunctionalGroupsSequence, including InstanceNumber
    out = deepcopy(ds)

    # --- Mark as simulated low-dose series ---
    if series_description is not None:
        out.SeriesDescription = series_description
    else:
        old_desc = str(getattr(ds, "SeriesDescription", "PCD")).strip()
        if "SIM" not in old_desc.upper():
            old_desc = old_desc.replace(" FD", "").strip()
            out.SeriesDescription = f"{old_desc} SIM {int(round(mAsFactor * 100))}pct"
        else:
            out.SeriesDescription = old_desc

    out.NumberOfFrames = n_frames
    out.Rows = H
    out.Columns = W

    # Keep original Rescale
    out.RescaleIntercept = intercept
    out.RescaleSlope = slope

    # PixelRepresentation only if present in FD
    if hasattr(ds, "PixelRepresentation"):
        out.PixelRepresentation = int(ds.PixelRepresentation)
    elif hasattr(out, "PixelRepresentation"):
        try:
            delattr(out, "PixelRepresentation")
        except Exception:
            if Tag(0x0028, 0x0103) in out:
                del out[Tag(0x0028, 0x0103)]

    # Root min/max: only if FD had them at root
    if TAG_SMALLEST in out:
        del out[TAG_SMALLEST]
    if TAG_LARGEST in out:
        del out[TAG_LARGEST]

    if had_root_smallest:
        out.add_new(TAG_SMALLEST, "US", int(pixel_u16.min()))
    if had_root_largest:
        out.add_new(TAG_LARGEST, "US", int(pixel_u16.max()))

    out.PixelData = pixel_u16.tobytes()

    # Update per-frame min/max and tube current only
    # Do NOT touch InstanceNumber — left as copied from FD multi-frame
    if (
        hasattr(out, "PerFrameFunctionalGroupsSequence")
        and out.PerFrameFunctionalGroupsSequence
    ):
        for i, item in enumerate(out.PerFrameFunctionalGroupsSequence):
            fmin = int(pixel_u16[i].min())
            fmax = int(pixel_u16[i].max())

            if TAG_SMALLEST in item:
                del item[TAG_SMALLEST]
            if TAG_LARGEST in item:
                del item[TAG_LARGEST]
            item.add_new(TAG_SMALLEST, "US", fmin)
            item.add_new(TAG_LARGEST, "US", fmax)

            if update_tube_current and TAG_TUBE_CURRENT in item:
                try:
                    old_ma = float(item[TAG_TUBE_CURRENT].value)
                    item[TAG_TUBE_CURRENT].value = int(round(old_ma * mAsFactor))
                except Exception:
                    pass
            # InstanceNumber (0020,0013) is left unchanged in this item

    if update_tube_current and hasattr(out, "XrayTubeCurrent"):
        try:
            out.XrayTubeCurrent = int(round(float(out.XrayTubeCurrent) * mAsFactor))
        except Exception:
            pass

    # New SOP UID per file
    out.SOPInstanceUID = generate_uid()
    if not hasattr(out, "file_meta") or out.file_meta is None:
        out.file_meta = Dataset()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.file_meta.MediaStorageSOPClassUID = out.get(
        "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2"
    )
    out.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    out.is_little_endian = True
    out.is_implicit_VR = False

    # Force US VR on any remaining min/max (per-frame and optional root)
    _force_minmax_us(out)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    dcmwrite(output_file, out, write_like_original=False)

    print(f"\n✅ Wrote noisy multi-frame CTPD ({n_frames} frames):")
    print(f"   {output_file}")
    print(f"   SeriesDescription = {out.SeriesDescription}")
    print(f"   RescaleIntercept (kept) = {intercept:.8g}")
    print(f"   RescaleSlope     (kept) = {slope:.8g}")
    print("   Per-frame InstanceNumber preserved")
    return output_file, projection3D


def insert_noise_multiframe_folder(
    input_dir,
    output_dir,
    mAsFactor,
    dictionary_path=None,
    Ne=0.0,
    seed=None,
    name_suffix="_noise",
    fineTuneFactor=1.0,
):
    """
    Apply multi-frame noise insertion to every .dcm in a folder.
    RescaleIntercept / RescaleSlope are preserved from each input file.
    Per-frame InstanceNumber is preserved.
    All output files share one SeriesDescription.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith((".dcm", ".ima"))
    )
    print(f"Found {len(files)} multi-frame files in {input_dir}")
    print("RescaleIntercept / RescaleSlope: KEEP ORIGINAL")
    print("Root PixelRepresentation / Smallest / Largest: only if present in FD")
    print("Per-frame InstanceNumber: PRESERVED")

    if not files:
        return []

    first = dcmread(os.path.join(input_dir, files[0]), force=True)
    old_desc = str(getattr(first, "SeriesDescription", "PCD")).strip()
    if "SIM" not in old_desc.upper():
        old_desc = old_desc.replace(" FD", "").strip()
    new_desc = f"{old_desc} SIM {int(round(mAsFactor * 100))}pct"
    print(f"SeriesDescription : {new_desc}")

    results = []
    for i, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        out_name = f"{base}{name_suffix}{ext}"
        out_path = os.path.join(output_dir, out_name)
        file_seed = None if seed is None else seed + i * 100000

        print(f"\n===== [{i + 1}/{len(files)}] {fname} =====")
        try:
            path, _ = insert_noise_multiframe(
                in_path,
                out_path,
                mAsFactor=mAsFactor,
                dictionary_path=dictionary_path,
                Ne=Ne,
                seed=file_seed,
                series_description=new_desc,
                fineTuneFactor=fineTuneFactor,
            )
            results.append(path)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n=== Finished: {len(results)}/{len(files)} files ===")
    return results


# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    dict_path = r"F:\PCD_DICOMCTPD_Generator\DICOM-CT-PD-dict_v11.txt"

    # --- One multi-frame file ---
    insert_noise_multiframe(
        input_file=r"F:\PCD_DICOMCTPD_Generator\Batch_00001_multiframe_ct_pd.dcm",
        output_file=r"F:\PCD_DICOMCTPD_Generator\Batch_00001_multiframe_ct_pd_25pct.dcm",
        mAsFactor=0.25,
        fineTuneFactor=1.0,  # e.g. 0.92 or 1.08 to match measured noise
        dictionary_path=dict_path,
        Ne=0.0,
        seed=42,
        update_tube_current=True,
    )

    # --- All multi-frame batches in a folder ---
    # insert_noise_multiframe_folder(
    #     input_dir=r"F:\PCD_DICOMCTPD_Generator\Test3\Tube1_Thr1",
    #     output_dir=r"F:\PCD_DICOMCTPD_Generator\Test3\Tube1_Thr1_noise25",
    #     mAsFactor=0.25,
    #     dictionary_path=dict_path,
    #     Ne=0.0,
    #     seed=42,
    #     name_suffix="_25pct",
    # )
