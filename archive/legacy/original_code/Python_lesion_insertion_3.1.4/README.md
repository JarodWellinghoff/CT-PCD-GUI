# DICOM-CT-PD Lesion Insertion Pipeline 3.1.4

This is a pure-Python replacement for the previous three-step MATLAB/Python pipeline. It provides one start-to-output GUI while retaining a JSON/command-line route for reproducible batch runs.

The revised pipeline:

- generates two-spectrum lesion models from T1/T2 reconstructed DICOM series plus an NRRD segmentation, including each lesion's original perilesional background HU;
- lets the user select insertion locations and measure T1/T2 background HU values interactively;
- reads scanner geometry and projection values directly from DICOM-CT-PD;
- forward-projects each lesion using the geometry stored in each projection header;
- creates an output DICOM-CT-PD series by replacing only `(7FE0,0010) PixelData`;
- does not require MATLAB, MATLAB Engine, `read_xacb10ext`, XML mode files, or proprietary raw writers.

## DICOM input organization

The pipeline supports both projection storage models:

- **single-frame**: one projection view per DICOM file;
- **multi-frame**: multiple sequential projection views in one DICOM file, with the count read from `(0028,0008) NumberOfFrames`.

The selected input may be one DICOM file or a common root directory containing multiple tube, threshold, series, or batch subdirectories. The directory is searched recursively and its hierarchy is reproduced in the output.

## Installation

Use Python 3.11 or newer. On Windows, open Anaconda Prompt in this project directory and run:

```powershell
conda env create -f environment.yml
conda activate dicom-ctpd-lesion-insertion
```

Alternatively:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Recommended use: one-step GUI

Run:

```powershell
python run_gui.py
```

The wizard has three pages.

### 1. Inputs

Select:

- **T1 reconstructed DICOM series**: low-threshold reconstructed images used to generate lesion channel 1 and select locations.
- **T2 reconstructed DICOM series**: high-threshold reconstructed images used to generate lesion channel 2 and measure the paired background.
- **Lesion segmentation**: Slicer-style `.nrrd` or `.seg.nrrd` file containing a segment named `Lesion`.
- **DICOM-CT-PD input**: one projection DICOM file or the root directory containing all single-frame or multi-frame instances/series.
- **Output directory**: a new or empty directory for models, configuration, summary, and modified projection files.
- **Mask alignment**:
  - `Physical NRRD/DICOM coordinates` uses NRRD space directions/origin plus DICOM Image Position/Orientation and is the recommended mode.
  - `Legacy flip/rotate alignment` reproduces the old hard-coded NRRD transformation and should only be used for older segmentations known to require it.
- **Projection workers**: start with `1` for validation. Increase after confirming a small case; each worker holds its own lesion-model copy in memory.

When you select **Next**, the program loads and checks T1/T2 alignment, maps the NRRD mask onto the DICOM grid, detects connected lesions, measures the channel-specific original background from the inverse segmentation mask inside each lesion bounding-box VOI, and writes both Python `.npz` and MATLAB-compatible `.mat` models.

### 2. Lesion placement

1. Move the slider to the desired slice.
2. Draw a rectangle on the T1 image over homogeneous liver parenchyma.
3. Review the measured mean T1/T2 HU values at the new insertion location.
4. Choose the lesion model to insert at that location.
5. Click **Add this placement**.
6. Repeat for additional lesions, avoiding overlap.

The program records the ROI center as `(column, row, zero-based slice)` and applies the same Alpha HFS/FFS coordinate conversion used by `lesion_insertion_alpha_HG_v2.m`. It corrects the selected position as `selected - ReconstructionTargetCenterPatient + DataCollectionCenterPatient`; FFS also reverses the projection-space x and z signs as in that MATLAB routine.

The placement ROI still records the **new-location** background HU for audit and expected-output reporting. It is not subtracted from the lesion model. The projector subtracts the **old** background stored when the model was generated.

### 3. Spectrum mapping and output

The program scans all DICOM-CT-PD headers and lists the detected spectrum and source indices. Map each spectrum to the corresponding lesion-model channel:

- spectrum/threshold 1 -> channel 1;
- spectrum/threshold 2 -> channel 2.

Confirm this mapping from your acquisition. The numeric spectrum index is read from private tag `(7033,1063)` in the supplied dictionary; the source index is read from `(7033,105D)`.

Click **Start lesion insertion**. The output DICOM files are written under:

```text
<selected-output>/dicom_ctpd_modified/
```

The input directory hierarchy and filenames are preserved.

## Command-line validation and batch use

Before a full run, inspect the projection data:

```powershell
python run_pipeline.py inspect --input "S:\path\to\DICOM_CTPD"
```

The result separately reports the number of DICOM files and the total number of projection frames, along with spectrum/source groups, detector dimensions, and stored Pixel Data orientation. If this command fails, do not run insertion; first check that the selected files use the supplied DICOM-CT-PD private tags.

For a reproducible batch run, edit [examples/insertion_config.example.json](examples/insertion_config.example.json), then run:

```powershell
python run_pipeline.py run --config examples\insertion_config.example.json
```

For each insertion, specify exactly one center representation:

- `center_pixel: [column, row, zero_based_slice]` together with `t1_series_dir`; or
- `center_ctpd_mm: [x, y, z]` if the position has already been converted to the DICOM-CT-PD coordinate system.

Lesion-model channels and `background_hu` use zero-based channel mapping internally. For example, `"1": 0` maps DICOM spectrum 1 to the first lesion channel. The configuration name `background_hu` is retained for compatibility; it contains the measured new-location T1/T2 backgrounds used for reporting, not for the projected subtraction.

## Outputs

```text
selected_output/
├── lesion_models/
│   ├── Lesion01.npz
│   ├── Lesion01.mat
│   └── ...
├── lesion_insertion_config.json
└── dicom_ctpd_modified/
    ├── <copied input hierarchy and modified DICOM files>
    └── lesion_insertion_summary.json
```

The JSON summary contains the converted CT-PD lesion centers, selected models, old and new background HU, original contrast, expected inserted mean HU, spectrum mapping, total and changed projection file/frame counts, number of changed pixels, number of clipped stored values, elapsed time, and per-file results.

## What is changed in each DICOM file

The physical projection is decoded as:

```text
line_integral = stored_uint16 * RescaleSlope + RescaleIntercept
```

For each lesion voxel and spectrum channel, the projected difference is:

```text
delta_mu = 0.01917 mm^-1 * (HU_lesion - HU_old_background) / 1000
```

The fixed coefficient is the unit-converted form of the MATLAB value `0.1917 cm^-1`. DICOM water-attenuation values are deliberately not read. Siddon ray tracing integrates `delta_mu` in mm^-1 over path length in mm. The added line integral is converted back with the original Rescale Slope/Intercept, rounded, and clipped to unsigned 16-bit range.

If the reconstructed image at the new site has mean `HU_new_background`, the expected mean after reconstruction is:

```text
HU_inserted = HU_lesion - HU_old_background + HU_new_background
```

Therefore the expected inserted contrast relative to the new background is the source contrast `HU_lesion - HU_old_background`.

For native, uncompressed Pixel Data, the program copies every source-file byte verbatim and replaces only values inside the existing Pixel Data byte range. Multi-frame files are memory-mapped and updated one frame at a time, avoiding a full floating-point copy of all frames in memory. The result is reread and all dataset and file-meta values except Pixel Data are compared with the source. If verification fails, that output file is rejected and removed. Files whose encoded pixels do not change remain byte-for-byte identical.

Compressed transfer syntaxes are intentionally rejected because writing modified compressed pixels would require changing the transfer syntax or recompressing the image. Decompress such a series before using this pipeline.

## Detector dimensions and stored Pixel Data orientation

The physical detector dimensions always come from the DICOM-CT-PD private tags:

- `(7029,1010) NumberofDetectorRows`;
- `(7029,1011) NumberofDetectorColumns`.

Standard DICOM `Rows (0028,0010)` and `Columns (0028,0011)` describe only the stored Pixel Data matrix. The pipeline accepts either of these layouts:

```text
private detector (rows, columns) == stored DICOM (Rows, Columns)
private detector (rows, columns) == stored DICOM (Columns, Rows)
```

For the second case—for example, a physical detector of `(144, 1376)` stored as a DICOM pixel matrix of `(1376, 144)`—the pipeline transposes the decoded projection into physical detector order before ray tracing and transposes it back before encoding. It does not change the original DICOM `Rows`, `Columns`, or file layout.

## Multi-frame geometry

Each frame is projected using its own scanner geometry. Version 3.1.4 recognizes frame-varying private values in any of these forms:

1. one top-level value per frame, where the value count equals `NumberOfFrames`; or
2. private geometry values stored in `(5200,9230) Per-frame Functional Groups Sequence`, including values nested within a functional-group item; or
3. a generator-specific sequence containing exactly `NumberOfFrames` items, with one acquisition/geometry item per projection frame.

Shared or scalar geometry may remain at the top level or in `(5200,9229) Shared Functional Groups Sequence`. The following values may vary by frame: detector focal-center angle, axial position and radial distance; source angular, axial and radial shifts; source index; spectrum index; and Rescale Slope/Intercept. Detector dimensions and storage orientation must remain constant within a DICOM instance.

Some generator-produced files have `NumberOfFrames` per-frame items but omit private acquisition values from the first item. If later items contain the tag, version 3.1.4 recovers a missing continuous scalar by frame-index interpolation or edge extrapolation; angular recovery follows the shortest step across an angle wrap. Categorical or non-scalar values use the nearest populated frame. Recovery is allowed only when the containing sequence has exactly `NumberOfFrames` items, and a `RuntimeWarning` reports every recovered tag and source frame. A tag absent from the entire sequence is not guessed.

## DICOM-CT-PD private tags used

| Purpose | Tag |
|---|---|
| Detector rows/columns | `(7029,1010)`, `(7029,1011)` |
| Detector element spacing | `(7029,1002)`, `(7029,1006)` |
| Detector shape | `(7029,100B)` |
| Detector focal-center angle/z/radius | `(7031,1001)`-`(7031,1003)` |
| Focal-center-to-detector distance | `(7031,1031)` |
| Central detector element | `(7031,1033)` |
| Source angular/axial/radial shifts | `(7033,100B)`-`(7033,100D)` |
| Source index | `(7033,105D)` |
| Spectrum index | `(7033,1063)` |
| Projection samples | `(7FE0,0010)` |
| Number of projection frames | `(0028,0008)` |
| Shared/per-frame functional groups | `(5200,9229)`, `(5200,9230)` |

The full supplied dictionary is retained as `docs/DICOM_CTPD_dictionary.txt`. The dictionary includes `(7041,1001) WaterAttenuationCoefficient`, but version 3.1.4 intentionally ignores that value and uses the fixed MATLAB coefficient described above.

## Changes from the prior pipeline

| Previous component | Replacement |
|---|---|
| `Step1_Lesion_Model_Generation.py` | GUI page 1 and `lesion_models.generate_lesion_models()` |
| `Step2_Lesion_Location_Selection.py` | GUI page 2 |
| `Step3_Lesion_Insertion.py` | GUI page 3 or JSON CLI |
| `raw_reader_Alpha.py` / MATLAB Engine | `lesion_pipeline/ctpd.py` |
| `writeAlphaRaw_Lesion.py` | DICOM Pixel Data encoder in `ctpd.py` |
| broken Python Siddon/geometry helpers | tested implementations in `projector.py` and `ctpd.py` |
| intermediate CSV editing | in-memory placement list plus saved JSON configuration |

Version 3.1.4 lesion models store `LesionMeanHUByChannel`, `OldBackgroundHU`/`LesionBackground`, `OldBackgroundMethod`, and `OldBackgroundVoxelCount`. Models from 3.1.3 used the superseded shell calculation and are intentionally rejected. Regenerate models from the T1/T2 reconstruction and NRRD segmentation so the inverse-mask background is measured rather than reused or guessed. New runs save `.npz` for unambiguous Python array shapes and `.mat` for MATLAB interoperability.

## Verification

Run the included synthetic tests:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

The tests cover private-tag decoding in explicit and implicit VR, single-frame and multi-frame Pixel Data, top-level geometry vectors, standard and generator-specific frame-aligned sequences, sparse first-frame metadata recovery, deferred/memory-mapped frame processing, direct and transposed Pixel Data storage, detector geometry, fixed water-coefficient behavior, old-background subtraction, inverse-mask bounding-box background measurement, Rescale Slope/Intercept round trip, Siddon path length, physical NRRD-to-DICOM alignment, `.mat` interoperability, end-to-end Pixel Data modification, and non-pixel header preservation.

The supplied 2,000-frame metadata dataset was used to validate discovery and geometry extraction for all frames, including its missing first-item private geometry. It did not contain accessible projection Pixel Data for a complete insertion/rewrite test. Before processing an entire patient scan:

1. run `inspect` on the real input;
2. copy a small contiguous subset of projections from both spectra;
3. run one small-lesion insertion with one worker;
4. confirm `clipped_pixels` is zero or scientifically explainable;
5. reconstruct and verify lesion location, orientation, contrast, and spectrum assignment;
6. only then run the complete series.

## Troubleshooting

- **No DICOM-CT-PD projection files found**: confirm the selected directory contains the private geometry tags listed above, not reconstructed CT images.
- **Missing `(7031,1001)` for frame 1**: use version 3.1.2 or newer. The affected generator layout leaves the first per-frame private-acquisition values empty; the pipeline recovers them from the next populated frame items and emits `RuntimeWarning` messages documenting the recovery.
- **Older lesion model is rejected**: regenerate it with version 3.1.4. The model must record that its old-background HU came from the inverse segmentation mask inside the lesion bounding-box VOI.
- **Bounding-box VOI has no unsegmented background voxels**: the segmentation completely fills the local VOI, so this method cannot estimate the original background. Revise the source segmentation/VOI definition or use a scientifically defined manual background ROI; the pipeline does not fall back to the whole CT.
- **T1/T2 slice positions do not match**: use corresponding reconstructed series with identical geometry.
- **Aligned NRRD lesion mask is empty**: verify NRRD space/origin/directions. Use legacy alignment only if the segmentation was created for the prior hard-coded transformation.
- **Header-preservation check failed**: retain the rejected input/output pair and inspect the transfer syntax and file-meta encoding before retrying.
- **Clipped pixels > 0**: the modified physical line integral fell outside the range representable by the source Rescale Slope/Intercept and uint16 Pixel Data. Do not ignore this warning.
- **Run is slow**: validate with one worker first, then increase workers according to RAM and CPU capacity. Workers parallelize DICOM files, not frames within the same file; one multi-frame file therefore uses one worker. Runtime scales with lesion size, affected detector rays, projection frame count, and number of inserted lesions.
- **Linux reports `libEGL.so.1` while starting the GUI**: install your distribution's EGL runtime package (for example `libegl1` on Debian/Ubuntu). This is a Qt system-library requirement; the command-line pipeline does not require a display.
