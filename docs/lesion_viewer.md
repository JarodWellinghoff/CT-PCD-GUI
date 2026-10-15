# Lesion Viewer: HU slices and file inspection

The **Views** control selects **2-D and 3-D**, **2-D only**, or **3-D only**.
The selected layout is remembered. The divider between 2-D and 3-D can be
resized, as can the file-information area below it. In 2-D-only mode, surface
rebuilds are deferred until a 3-D view is requested.

## Original stored values

For the lesion-library schema, the 2-D source is **VOI**, not **LesionMask**.
No HU offset, contrast normalization, resampling, or mask multiplication is
applied to the stored values. Window/level changes only the grayscale display.
Hover or left-click to read the stored voxel value and its one-based slice,
row, and column. The readout follows slice changes. It is not a measurement
from the displayed 8-bit image.

Use the shared viewer controls: wheel to scroll slices, slider or number box
to jump to a slice, Ctrl+wheel to zoom, middle-drag to pan, right-drag to change
window/level, or the window-preset menu. Fit and Reset view remain available.
Voxel aspect ratio follows the configured X/Y spacing. The existing axis-order
control governs the stored array interpretation; these views do not invent
patient-orientation information absent from the archive.

## Two thresholds / multiple reconstructed series

The documented multi-series NPZ format uses **VOI[row, column, slice, channel]**.
Each channel appears in its own view, labelled from `SeriesNames` when available.
In **2-D only** mode, the threshold/channel views are **side-by-side**. In
**2-D and 3-D** mode, they are **vertically stacked** beside the 3-D surface.
Switching layouts preserves the current slice, outline selection, and each
channel's window/level settings. Changing the slice in any view updates all views
to the same image number. Window/level remains independent for each channel.
The loader uses the documented final channel axis, never the shortest spatial
axis; single-slice and singleton-channel NPZ files are supported.

## Lesion outline

The **Lesion outline** checkbox overlays the stored `LesionMask` boundary.
It supports a shared 3-D mask, a singleton-channel mask, or one mask per channel.
The overlay does not alter HU values and is independent of the 3-D smoothing,
threshold, and mesh-reduction controls. Missing or mismatched masks are reported
rather than resampled or silently substituted.

A mask-only or unrecognized legacy archive cannot supply original HU values.
For those files the selected 3-D array remains inspectable in 2-D, explicitly
labelled **Stored value**, not HU. The absence of original HU is stated in the
viewer. Numeric MATLAB arrays and readable MATLAB metadata remain supported;
MATLAB v7.3 channel-first HDF5 storage is normalized when its MATLAB marker is
present, retaining the existing reader's configured spatial ordering.

## File information

The summary retains the selected-array and surface metrics. The **File metadata /
DICOM headers** tab exposes available stored lesion statistics, geometry,
source paths, series identifiers, per-channel statistics, decoded JSON headers,
and array shape/type summaries. Use **Find next** to search, or **Copy metadata**
to copy this local inspection report. Large arrays are summarized rather than
printed in full; sampled statistics and truncation are identified. Stored
metadata can contain patient information, so handle copied reports accordingly.

Object/pickled NPZ members remain blocked. Readable fields are still shown and
skipped members are reported in warnings. Loading another file clears old
images, metadata, and meshes; superseded surface results are ignored.

**Save screenshot** captures the currently selected 2-D/3-D layout. Mesh export
continues to export the current 3-D surface.

## Tests

Install the repository's `dev` dependencies, then run:

```bash
python -m pytest tests/unit/test_lesion_inspection.py tests/gui/test_lesion_slice_views.py
```

The unit fixtures are synthetic and include single/dual channels, singleton
spatial dimensions, geometry permutations, precision, masks, metadata, blocked
pickle members, and MATLAB loading. The GUI tests cover both navigation
controls, side-by-side and stacked views, state preservation on mode changes,
probes, outlines, restored layout preferences, screenshots, and file clearing. Workspace-layout tests substitute a QWidget for the VTK scene; a real
application smoke test is still required for the native VTK/Qt integration.
