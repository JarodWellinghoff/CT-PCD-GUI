# FreeCT Reconstruction module

This module integrates the **Go-based FreeCT v2 `fct-wfbp` interface** found in
`freect-v2.0.2-ZY.zip`. It is not an adapter for the older CUDA `FreeCT_wFBP`
parameter-file interface. FreeCT remains a separate executable; this change
neither redistributes its binaries nor changes its reconstruction algorithm.

**Research use only.** A successful process and readable volume do not establish
correct geometry, HU calibration, image orientation, or diagnostic suitability.
Validate against a known phantom/reference reconstruction before using results
in research measurements. No patient-space registration is inferred.

## Installation and engine selection

Install the updated application from the repository root:

```console
python -m pip install -e ".[dev]"
ct-pcd-gui
```

PyYAML is the only added Python dependency. The existing NumPy, pydicom and
PySide6 dependencies are reused. No nibabel, SimpleITK, CUDA, or GPU dependency
is added by this adapter. Requirements of the selected FreeCT executable still
apply independently.

On Windows, extract the supplied archive and select the trusted `.exe` in its
`cmd/fct-wfbp` directory that you normally use. There are multiple customized
executables: the module does not guess which one is appropriate or assume all
of them match the included source. On Linux/macOS select a compatible native
executable built for that platform; this module does not run Windows binaries
through WSL or Wine. An installed `fct-wfbp` on PATH is also accepted.

The command contract is:

```console
fct-wfbp --no-telemetry generate-base-config --input-battelle=STAGING_DIRECTORY
fct-wfbp --no-telemetry recon CONFIG_FILE
```

Arguments are passed separately without a shell, including paths with spaces.
`--no-telemetry` precedes the subcommand because the supplied executable checks
it during startup. Only the executable path is remembered in Qt settings;
patient input/output paths are not persisted there.

## Workflow

1. Open **FreeCT Reconstruction** in the module selector. Select the executable,
   a CTPD projection folder or one multi-frame CTPD file, and a working/output
   folder **outside** the input tree. Use one study/series per input. For a
   multi-spectrum or multi-source acquisition, set the actual DICOM spectrum
   and source indices. `Auto` accepts only an unambiguous single group.
2. Select **Prepare data and configuration**. Preparation reads headers,
   stages ordered single-frame projections in a new job directory, and asks
   FreeCT to derive scan geometry. Original inputs are never modified. Inspect
   the generated geometry and preprocessing in the advanced configuration tab,
   especially indexing correction, detector flips and tube-angle negation.
3. Set the reconstruction start/end Z, field of view, matrix, slice thickness,
   slice spacing, filter and optional HU conversion in the panel. The panel's
   values override the corresponding YAML fields. Advanced geometry and other
   YAML fields remain editable. Managed input/output paths are always rebound
   to the current job. Do not guess a missing HU calibration coefficient.
4. Select **Reconstruct images**. Work runs through the shared Qt task runner,
   with a live status/log pane and cancellation. The progress bar is determinate
   for staging/output validation and indeterminate while the engine is running;
   engine progress text remains visible. Cancellation stops the native process
   tree and retains partial files, which are not presented as completed images.
5. Inspect completed slices with the existing shared viewer controls. HU output
   supports its standard window/level presets. Non-HU output is clearly labelled
   and auto-scaled per slice rather than incorrectly applying HU presets. Matrix
   dimensions may differ; the preview preserves physical pixel aspect for the
   configured square field of view. Patient left/right and LPS/RAS orientation
   are deliberately not asserted.

A new reconstruction attempt is created each time, even when reusing prepared
projections. Completed volumes are not overwritten by a later run.

## Outputs and storage

```text
OUTPUT/freect-TIMESTAMP-ID/
  projections/projection_000000000.dcm ...
  projection_manifest.json
  generate-config.stdout.log
  generate-config.stderr.log
  generated-config.yaml
  status.json
  recon-TIMESTAMP-ID/
    reconstruction.yaml
    recon.stdout.log
    recon.stderr.log
    freect_native.nii.gz
    reconstruction.nii
    result.json
    status.json
    dicom_native/                 # only when explicitly requested
```

`reconstruction.nii` is the validated, normalized float32 NIfTI-1 volume used
for preview. `freect_native.nii.gz` is the engine's unchanged original output.
`result.json` records scanner-grid slice positions, spacing, intensity units,
configuration/executable references and the absence of patient orientation.
An attempt is complete only when its `status.json` says `completed`.

**Staged projections, configurations, logs and manifests may contain patient
identifiers and source paths. This is not an anonymizer.** Use approved local
storage and access controls. No projection upload is implemented. Staging and
all attempts remain on disk, including cancelled/failed work; remove obsolete
job directories manually only after they are no longer running. Staging can
require approximately an additional uncompressed copy of the selected scan.
One source object is decoded at a time, but a very large multi-frame object
still needs enough RAM for that entire object's decoded pixel array. FreeCT
has its own additional memory requirements.

## Input contract and explicit limits

The adapter supports little-endian 16-bit integer, single-component CTPD
PixelData. Compressed data requires an available pydicom decoder. Big-endian
private geometry is rejected rather than relabelled. Detector dimensions must
match stored dimensions directly or transposed; the native reader, not the GUI,
handles the existing CTPD transpose convention. Encoded pixels and each frame's
rescale values are preserved, so FreeCT applies rescaling exactly once.

Multi-frame functional groups must contain per-projection angles and the
correct number of items. Shared and nested per-frame geometry/rescale values
are flattened into standalone staging files. Global `InstanceNumber` defines
projection order, not filename order. A single multi-frame object can fall back
to stored frame order; chunked objects cannot safely use that fallback. Duplicate
numbers, irregular gaps, changing detector dimensions and mixed study/series or
spectrum/source groups are rejected. A constant instance-number stride is
allowed for an explicitly selected interleaved spectrum. Missing indices are
not assumed to be 1 when the metadata declares multiple spectra or sources.
This validation cannot detect every possible missing or geometrically incorrect
projection: inspect the acquisition and the generated configuration.

The guarded contract is cylindrical detector, FANBEAM projection geometry,
HELICAL or AXIAL acquisition and FFSNONE (no flying focal spot). It requires at
least 10 staged projections and 10 detector rows because the supplied source's
progress reporting divides by zero for smaller loops. Projections per rotation
must be even, at least four, and no greater than the staged count. The GUI
supports matrices 16 through 4096 per axis and at most 32767 slices. Adaptive
filtration is rejected because it is unimplemented in the supplied source.
These restrictions intentionally do not promise support for every scanner or
for every customized binary in the archive.

## Supplied-source compatibility corrections

These are based on inspection of the uploaded source, not on an assumption that
any particular prebuilt `.exe` contains the same code:

- The CLI initializes its configuration override struct before command dispatch;
  some builds consequently generate zero filter coefficients. Generated zero
  `KernelA`/`KernelC` values are replaced by the source's 0.5/0.5 defaults, with
  log messages, and remain editable. Both coefficients are serialized as YAML
  floats to satisfy Go's `float64` assertions, including a ramp coefficient 1.0.
- The base-config generator's stationary-table calculation can yield NaN start
  and end positions. An axial acquisition defaults to one central slice at the
  finite table position, for explicit user review rather than guessing coverage.
- The included NIfTI writer creates a 348-byte NIfTI-1 header with the `n+2`
  signature and writes slice thickness where inter-slice spacing belongs. The
  separate normalized volume corrects the signature only for a verified
  NIfTI-1/float32 layout and sets spacing from the executed configuration. It
  clears unsupported qform/sform claims rather than fabricating a patient-space
  transform. Voxel values are unchanged; the native file is retained.
- FreeCT can log output/HU-conversion errors yet exit with status zero. The
  adapter checks error logs, expected dimensions, finite voxel values, exact
  payload length and gzip integrity before publishing a result.

The optional **native DICOM** output is disabled by default and experimental.
The included writer has incomplete metadata and a constant SOP Instance UID.
This adapter does **not** repair or certify those DICOM files; it only requests
and checks their presence/count. Do not upload them to PACS, use them for
clinical decisions, or treat them as a valid registered DICOM CT series. The
primary supported result is the normalized research NIfTI volume and sidecar.

## Tests and acceptance checks

```console
python -m pytest tests/unit/test_freect_backend.py tests/unit/test_freect_dicom.py tests/gui/test_freect_module.py
```

Backend tests exercise configuration contracts, actual subprocess invocation and
cancellation, corrupt/synthetic NIfTI data, and orchestration with a simulated
engine. DICOM tests construct synthetic single- and multi-frame projections to
check ordering, splitting, private metadata, rescale preservation, input
immutability, grouping and rejection cases. Qt tests exercise module state,
fast-task signal delivery, cancellation and viewer behaviour. They require
pydicom/PySide6/pytest-qt and skip at collection when these dependencies are absent.

Before accepting a deployment, run those tests in the application's environment,
then reconstruct a known phantom/reference with the selected native FreeCT
executable. Check matrix and slice positions, coverage, detector/angle direction,
left/right orientation, HU accuracy and voxel spacing against the reference.
Compare single-frame and equivalent multi-frame inputs, exercise cancellation,
and test paths containing spaces. Synthetic adapter tests do not validate the
reconstruction algorithm or scanner-specific calibration.
