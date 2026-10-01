# Lesion insertion module

> **Research use only.** This module does not make clinical-validation claims. The image-domain preview is an approximation; final lesion insertion is performed in DICOM-CT-PD projection data by the bundled pipeline 3.1.4.

## Workflow

1. Select a reconstructed DICOM file or directory. The module reads DICOM metadata, groups images by `SeriesInstanceUID`, and lets the user select one series.
2. Select the associated DICOM-CT-PD source. The module inspects projection geometry and compares available patient, study, frame-of-reference, and patient-position identifiers without writing patient identifiers to logs or manifests.
3. Select the MAT/NPZ lesion library. Each record is validated with the pipeline 3.1.4 lesion-model loader. Malformed or incompatible records remain visible with a warning.
4. Browse slices and click a placement location. The viewer maintains all three coordinate representations:
   - viewer voxel: `(column, row, zero-based slice)`;
   - DICOM patient coordinates: LPS millimetres using `ImageOrientationPatient`, `ImagePositionPatient`, and non-square `PixelSpacing`;
   - validated Alpha projector coordinates: HFS/FFS CTPD millimetres.
5. Add one or more lesions. Each instance has independent position, contrast, X/Y/Z scale, X/Y/Z rotation, per-channel background HU, preview opacity, visibility, and enabled state.
6. Review the debounced image-domain preview. Before, overlay, and approximate-after modes are available. The preview never replaces the final raw-data operation.
7. Validate, choose a new output location, and run final processing. The module stages transformed copies of lesion models, invokes the coworker's forward projector, and atomically promotes the result.
8. Optionally configure an external reconstruction command. The repository does not contain a reconstruction engine; the command can use `{input}`, `{output}`, and `{session}` placeholders.

## Architecture

The feature follows the existing module registry and presentation/application/domain/infrastructure split:

- `domain/models.py` contains immutable lesion instances, session serialization, validation issues, and centralized coordinate transforms.
- `application/session.py` manages independent multi-lesion state.
- `application/validation.py` creates the preflight validation summary.
- `infrastructure/dicom_service.py` discovers, orders, loads, and associates DICOM data.
- `infrastructure/lesion_library.py` adapts the connected local MAT/NPZ library.
- `infrastructure/model_transform.py` creates temporary contrast/scale/rotation variants without modifying source lesions.
- `infrastructure/preview.py` generates and caches the approximate image-domain preview.
- `infrastructure/legacy_adapter.py` wraps pipeline 3.1.4 for final DICOM-CT-PD processing and creates the reproducibility manifest.
- `infrastructure/reconstruction.py` runs an optional external reconstruction executable without a shell.
- `presentation/` contains Qt widgets and the presenter. Expensive I/O and computation use the shared `QtTaskRunner`.
- `_vendor/lesion_pipeline/` is a package-local copy of the coworker's pipeline core. The algorithm is not duplicated in UI event handlers.

## Spatial and data-integrity policy

The viewer supports arbitrary DICOM orientation for patient-coordinate display and placement. Final insertion is deliberately blocked when the source pipeline's validated coordinate assumptions are not met. In particular, final processing currently requires:

- `PatientPosition` equal to `HFS` or `FFS`;
- axis-aligned axial reconstruction orientation;
- a uniform slice grid without duplicate or missing-slice gaps;
- a complete spectrum-to-model-channel map;
- compatible reconstruction and projection identifiers;
- a new or empty output directory outside every input tree.

This is safer than silently approximating oblique or nonuniform data. The module reports missing `ReconstructionDiameter`, `ReconstructionTargetCenterPatient`, and `DataCollectionCenterPatient` when pipeline 3.1.4 fallback values would be used.

Source DICOM and lesion files are opened read-only. Final DICOM-CT-PD output is written under `dicom_ctpd_modified/`; source paths are never output targets.

## Reproducibility output

A successful run writes:

- `dicom_ctpd_modified/` — modified projection DICOM and the pipeline summary;
- `lesion_models_used/` — transformed per-instance lesion models used by the projector;
- `lesion_insertion_config.json` — resolved projector configuration;
- `lesion_insertion_session.json` — reopenable GUI session;
- `lesion_insertion_manifest.json` — input series identifiers, coordinate conventions, lesion sources and transforms, algorithm/application versions, spectrum mapping, worker count, and output paths.

Patient names, birth dates, and raw patient IDs are not included in the manifest.

## External reconstruction command

The optional command is tokenized and executed directly, not through a command shell. Example:

```text
C:\Tools\reconstruct.exe --input "{input}" --output "{output}" --session "{session}"
```

`{input}` is the final modified DICOM-CT-PD directory, `{output}` is the selected reconstruction-output directory, and `{session}` is the saved session JSON. The output directory must be new or empty and separate from all inputs and the raw-output tree.

## Development commands

```bash
python -m pip install -e ".[dev]"
ct-pcd-gui
pytest
ruff check .
pyright
```

Targeted tests:

```bash
pytest tests/unit/test_lesion_insertion_*.py
```

## Manual test checklist

1. Launch `ct-pcd-gui`, select **Lesion Insertion**, and verify the research-use and approximate-preview labels.
2. Drag a mixed DICOM directory into Reconstruction, click **Discover**, and confirm series are grouped by metadata rather than filename.
3. Select a valid series and verify physical slice order, mouse-wheel navigation, Ctrl+wheel zoom, middle-button pan, right-drag window/level, presets, fit, and reset.
4. Load associated CTPD data and verify the displayed file/frame/spectrum counts and association result.
5. Intentionally choose mismatched data and verify final processing is blocked with an actionable message that does not reveal patient values.
6. Scan the real lesion library, search/filter it, inspect metadata and thumbnails, and verify malformed records show a warning without terminating the scan.
7. Click a 3-D location, add a lesion, and confirm crosshair, selected marker, voxel coordinates, and table selection agree.
8. Change contrast, anisotropic scale, rotations, opacity, and position. Verify the preview becomes stale/calculating and then updates without freezing the UI.
9. Toggle before/overlay/after and preview on/off. Verify the preview remains labeled approximate.
10. Add at least two lesions. Select and edit each independently; duplicate, hide, disable, reorder, undo, redo, and remove them.
11. Save the session, close/reopen it, reload the JSON, and verify lesions and processing settings are restored.
12. Attempt final output in an input directory, inside an input tree, and in a non-empty directory; verify each is rejected.
13. Run final insertion to a new directory, observe progress/logs, and test cancellation. For a large multi-frame file, confirm the UI explains that cancellation occurs at a projection-file boundary.
14. Verify source files are byte-for-byte unchanged and output DICOM headers are preserved except for intended Pixel Data changes.
15. Inspect the manifest, configuration, session, transformed lesion models, and pipeline summary. Confirm staged temporary paths were relocated to final paths.
16. When an approved reconstruction executable is available, configure the command, run it, and compare the reconstructed result with the approximate preview. Domain experts must validate spatial placement and scientific correctness using representative real data.
