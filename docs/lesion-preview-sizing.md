# Compact lesion-preview height correction

This is an incremental update on top of the working lesion-library-preview patch.
It changes the compact source preview, not the target reconstruction viewer,
lesion values, insertion parameters, or threshold routing.

## Controls

A **2-D image height** control above the preview defaults to **140 px**, with a
range of **64-320 px**. It changes both source-threshold canvases together. Lower
it to 96 or 112 for a smaller preview. The setting stays in effect while selecting
other lesions during this module instance; it is not saved across application
restarts. Change `self.image_height.setValue(140)` to change the startup default.

Images are top-aligned and fitted with their physical aspect ratio preserved.
There may still be unused space below or beside an image whose aspect ratio does
not match the canvas. This update does not crop black pixels from the source VOI,
zoom into the segmentation, or stretch the image just to fill the viewport.
The existing Fit button restores the full image after a deliberate zoom or pan.

The 2-D tab sizes around the actual canvases, toolbar, navigation, and wrapped
readouts. Its inner scroll area is removed from the layout; the existing insertion
panel supplies scrolling. It no longer inherits the hidden 3-D page's preferred
height. The 3-D tab retains an independent 300-px height and lazy rendering.

The shared `HuImageView.set_qimage` also installs the spacing-scaled scene bounds
before the first fit and never temporarily resets them to raw pixel bounds on
slice/window-level refresh. Full-size viewers keep their existing layout sizes.

## Installation

Close the application. From the repository root, with the earlier preview already
installed:

```bash
git apply --check lesion-preview-height-fix.patch
git apply lesion-preview-height-fix.patch
```

Alternatively, replace the two Python files with those at the same paths in
`lesion-preview-height-fix-files.zip`. That ZIP includes only these corrected
source files, a regression test, and this note; it is not the full application.
Back up local edits first, and do not reapply the original preview patch.

## Validation

Run in the project's development environment (PySide6 and pytest-qt installed):

```bash
python -m pytest -q tests/gui/test_library_preview_sizing.py tests/gui/test_library_lesion_preview.py tests/gui/test_lesion_slice_views.py
```

The new tests check canvas and tab shrinking, top alignment, nested-scroll removal,
control visibility, 2-D/3-D height isolation, reload/cleanup, unchanged source HU,
synchronized navigation, and physical scene bounds at submillimetre and greater
than 1-mm spacing. The supplied fix passed syntax, whitespace, and clean patch
application checks in the preparation environment. GUI tests could not execute
there because PySide6 is unavailable; interactive rendering is not claimed tested.

On the workstation, check both thresholds at the usual dock width, adjust the
height, scroll slices, change window/level, use Fit and zoom/pan, and open/close the
3-D tab. At narrow dock widths the outer panel should scroll rather than clipping
the source preview's navigation or HU readout.
