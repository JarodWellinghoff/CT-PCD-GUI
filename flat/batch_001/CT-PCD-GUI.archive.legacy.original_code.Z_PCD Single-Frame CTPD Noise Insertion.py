"""
PCD DICOM-CT-PD projection-domain noise insertion (Python) — single-frame

Algorithm (same as MATLAB PCD_Noise_Insertion):
  P_A   = log-normalized attenuation projection (from CTPD + Rescale)
  N1    = incident NEPN profile  (tag PhotonStatistics, already mA-scaled)
  N1_det = N1 * exp(-P_A)                 # full-dose detected counts
  N2_det = N1_det * mAsFactor             # reduced-dose detected counts
  P_B   = P_A + sqrt( (1/N2 - 1/N1)*(1 + Ne/N2 + Ne/N1) ) * randn

RescaleIntercept / RescaleSlope are kept from the original FD CTPD file.
Pixel values are re-encoded with those fixed tags (with clipping to [0, 65535]).

Usage:
  insert_noise_single_file(...)
  or
  insert_noise_folder(...)
"""

import os
import glob
from pathlib import Path
import numpy as np
from pydicom import dcmread, dcmwrite
from pydicom.tag import Tag
from pydicom.uid import generate_uid
from copy import deepcopy
from src.ct_pcd_gui.features.noise_insertion.domain.noise_model import (
    add_poisson_noise_log_domain,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
TAG_PHOTON_STATS = Tag(0x7033, 0x1065)
TAG_SMALLEST = Tag(0x0028, 0x0106)
TAG_LARGEST = Tag(0x0028, 0x0107)


def _read_photon_statistics(ds):
    """Return PhotonStatistics as 1-D float32 array (incident NEPN profile)."""
    if TAG_PHOTON_STATS not in ds:
        raise KeyError("PhotonStatistics (7033,1065) not found in DICOM header.")

    val = ds[TAG_PHOTON_STATS].value
    if isinstance(val, (bytes, bytearray)):
        return np.frombuffer(val, dtype=np.float32).copy()
    return np.asarray(val, dtype=np.float32).ravel()


def _get_attenuation(ds):
    """Read pixel data and apply Rescale → attenuation projection P_A."""
    raw = ds.pixel_array.astype(np.float64)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        return raw * slope + intercept
    return raw


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


# ----------------------------------------------------------------------
# Single-file noise insertion
# ----------------------------------------------------------------------
def insert_noise_single_file(
    input_file,
    output_file,
    mAsFactor,
    Ne=0.0,
    seed=None,
    update_tube_current=True,
):
    """
    Insert noise into one single-frame PCD DICOM-CT-PD file and save result.
    RescaleIntercept and RescaleSlope are kept from the input (FD) file.
    """
    rng = np.random.default_rng(seed)

    ds = dcmread(input_file, force=True)

    # 1. Attenuation projection
    P_A = _get_attenuation(ds)

    # 2. Original Rescale (must keep)
    if not (hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept")):
        raise KeyError("RescaleSlope / RescaleIntercept not found in input CTPD.")
    intercept = float(ds.RescaleIntercept)
    slope = float(ds.RescaleSlope)

    # 3. Incident NEPN profile
    N1 = _read_photon_statistics(ds)

    # 4. Add noise
    P_B = add_poisson_noise_log_domain(P_A, N1, mAsFactor, Ne=Ne, rng=rng)

    # 5. Encode with fixed (original) Rescale
    pixel_u16, n_clipped = _encode_with_fixed_rescale(P_B, intercept, slope)

    # 6. Build output dataset
    out = deepcopy(ds)
    out.PixelData = pixel_u16.tobytes()

    # Keep original Rescale
    out.RescaleIntercept = intercept
    out.RescaleSlope = slope

    # Update min/max of stored pixels (force VR=US for Explicit VR safety)
    if TAG_SMALLEST in out:
        del out[TAG_SMALLEST]
    if TAG_LARGEST in out:
        del out[TAG_LARGEST]
    out.add_new(TAG_SMALLEST, "US", int(pixel_u16.min()))
    out.add_new(TAG_LARGEST, "US", int(pixel_u16.max()))
    out.PixelRepresentation = 0

    # Optional: reflect reduced dose in tube current
    if update_tube_current and hasattr(out, "XrayTubeCurrent"):
        try:
            out.XrayTubeCurrent = int(round(float(out.XrayTubeCurrent) * mAsFactor))
        except Exception:
            pass

    # New SOP Instance UID
    out.SOPInstanceUID = generate_uid()
    if hasattr(out, "file_meta") and out.file_meta is not None:
        out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID

    out.is_little_endian = True

    dcmwrite(output_file, out, write_like_original=False)
    return output_file, n_clipped


# ----------------------------------------------------------------------
# Folder batch processing
# ----------------------------------------------------------------------
def insert_noise_folder(
    input_dir,
    output_dir,
    mAsFactor,
    Ne=0.0,
    seed=None,
    file_suffix=None,
    update_tube_current=True,
):
    """
    Process all single-frame .dcm/.ima files in input_dir.
    RescaleIntercept / RescaleSlope are preserved from each input file.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith((".dcm", ".ima"))
    )
    if not files:
        print(f"No DICOM files found in {input_dir}")
        return []

    print(f"Noise insertion: mAsFactor = {mAsFactor}")
    print(f"RescaleIntercept / RescaleSlope: KEEP ORIGINAL")
    print(f"Input  : {input_dir}  ({len(files)} files)")
    print(f"Output : {output_dir}")

    out_paths = []
    total_clipped = 0

    for i, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)

        if file_suffix:
            base, ext = os.path.splitext(fname)
            out_name = f"{base}{file_suffix}{ext}"
        else:
            out_name = fname

        out_path = os.path.join(output_dir, out_name)
        file_seed = None if seed is None else seed + i

        try:
            path, n_clipped = insert_noise_single_file(
                in_path,
                out_path,
                mAsFactor=mAsFactor,
                Ne=Ne,
                seed=file_seed,
                update_tube_current=update_tube_current,
            )
            out_paths.append(path)
            total_clipped += n_clipped
        except Exception as e:
            print(f"  FAILED {fname}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  Processed {i + 1}/{len(files)}")

    print(f"Done. Wrote {len(out_paths)} files.")
    if total_clipped > 0:
        print(
            f"Note: {total_clipped} pixel(s) were clipped to [0, 65535] "
            f"when encoding with the original Rescale range."
        )
    else:
        print("No pixels were clipped when encoding with the original Rescale range.")
    return out_paths


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Single file ---
    # insert_noise_single_file(
    #     input_file=r"F:\PCD_DICOMCTPD_Generator\PCD_Batch_00001_FD\frame_00001.dcm",
    #     output_file=r"F:\PCD_DICOMCTPD_Generator\PCD_Batch_00001_noise25\frame_00001.dcm",
    #     mAsFactor=0.25,
    #     seed=42,
    # )

    # --- Whole folder ---
    base_dir = r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Full_dose_CTPD_File"
    dcm_files = glob.glob(os.path.join(base_dir, "**", "*.dcm"), recursive=True)
    input_dirs = list(set([str(Path(f).parent) for f in dcm_files]))
    input_dirs.sort()
    output_dirs = [
        f.replace("Full_dose_CTPD_File", "Quarter_dose_CTPD_File_CLI")
        for f in input_dirs
    ]
    for input, output in zip(input_dirs, output_dirs):
        insert_noise_folder(
            input_dir=input,
            output_dir=output,
            mAsFactor=0.25,  # quarter dose
            Ne=0.0,
            seed=42,
            file_suffix="_noise",  # optional name tag
            update_tube_current=True,
        )
