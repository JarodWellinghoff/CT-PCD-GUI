# Lesion insertion: source-library preview

Selecting a compatible library lesion now opens a compact **2-D HU** preview in
section 2 instead of the binary maximum-intensity-mask thumbnail. Each source
threshold has its own viewer. Thresholds are side by side and share the same
slice index. The controls are the existing lesion viewer's `LesionSliceViews`
and shared `DicomSliceViewer`: scroll/slider/spinbox navigation, window/level,
fit/reset, zoom/pan, a stored-HU pixel probe, and an optional lesion outline.

**3-D shape** uses the existing `VtkSurfaceBuilder` and `LesionVtkScene`. The mesh
and OpenGL scene are created on demand, not when browsing in 2-D. Its source
threshold selector is strictly an inspection control: it does not change the
source or target threshold choices for a placed lesion. The mesh shows the
selected source mask, including disconnected components, with no smoothing or
decimation. Empty masks or singleton spatial dimensions show an explanation;
2-D inspection remains available. Rendering/import failures also leave 2-D usable.

The information section can be collapsed. **Summary** contains file/lesion
identity, source threshold count, VOI shape and physical extent, voxel spacing,
segmented voxel count and mask volume per threshold, finite masked-HU statistics,
and the available stored background/contrast and acquisition information.
**Model metadata** contains bounded, selectable JSON for the loaded insertion
model. It is not advertised as a complete original DICOM-header export.

## Data and lifecycle

The existing insertion loader supplies `voi_hu` and `mask`. HU arrays are kept as
read-only views with no normalization, resampling, contrast adjustment or
placement transform. They represent the loaded source model, not the target
scan and not the approximate or final insertion result. The loader's explicit
(row, column, slice) order is passed as `YXZ`; display/mesh spacing is explicitly
reordered to (column, row, slice), i.e. XYZ. Reported volumes use the original
mask voxel count, not an approximate smoothed-mesh volume. Non-finite masked HU
values are counted and excluded from statistics, not silently converted to zero.

Preview-specific panel/presenter extensions are wired at the module factory.
The original insertion/session/threshold code stays unchanged. The enhanced
presenter bypasses the old thumbnail task, so there is no second thumbnail load.
Workers and VTK mesh construction use the shared task runner. Request generations
guard successes, failures and cancellation callbacks; clearing the selection or
filtering to no matches immediately removes the previous data. Superseded workers
remain included in busy tracking until they finish. Module cleanup invalidates
pending results and finalizes an instantiated VTK scene.

## Validation

Run in the repository's development environment:

```bash
pytest -q tests/unit/test_library_preview_data.py tests/gui/test_library_lesion_preview.py
```

The focused data tests cover lossless HU views, one/multiple thresholds, per-channel
and shared masks, RCS/XYZ geometry, exact statistics, non-finite values, empty and
single-slice masks, invalid shapes/spacing, invalid channel selection, and
cancellation. Offscreen GUI tests cover synchronized navigation, pixel values,
metadata, clear/reload, lazy scene creation, scene cleanup, stale load/error/mesh
results, empty filtered selections, and module-factory wiring. The scene is replaced
only in the 3-D UI-delegation test; this is not a GPU/OpenGL rendering test.

Before merging, check the GUI with representative MAT/NPZ lesions on the target
workstation: one and two thresholds; readable controls at the usual dock width;
HU/outline alignment; 3-D rotation/zoom/reset and threshold switching; rapid library
selection/filtering; and closing while a preview is loading. Confirm that preview
controls leave the placed lesion's source/target threshold selections unchanged.
