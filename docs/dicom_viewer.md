# Reusable reconstructed-DICOM viewer

`ct_pcd_gui.shared.qt.dicom_viewer` provides the standard controls for any
module that displays reconstructed DICOM slices. The lesion insertion and lesion
extraction modules both use this component.

## Standard interaction

- **Fit** or `F`: fit the current image to the viewport.
- **Reset view** or `R`: reset zoom/pan and restore the configured reset window.
- Mouse wheel: move through slices.
- `Ctrl` + mouse wheel: zoom around the pointer.
- Middle-button drag: pan.
- Right-button drag: adjust window width and level.
- The slice slider and spin box stay synchronized. The public API and emitted
  slice index are zero-based; the spin box is one-based for display.

The built-in presets are Soft tissue, Liver, Lung, and Bone. A manually adjusted
window is shown as **Custom**. Modules can preserve a DICOM-provided custom
window as the reset value with `remember_for_reset=True`.

## Components

`DicomImageView` is the reusable `QGraphicsView` canvas. It owns image display,
zoom, pan, slice-wheel, click, and window/level mouse behavior.

`DicomSliceViewer` wraps a `DicomImageView` with the common toolbar and slice
navigation controls. It can display a windowed HU array with `set_hu_image()` or
an already rendered grayscale/RGB/RGBA array with `set_rgb_image()`.

```python
from ct_pcd_gui.shared.qt.dicom_viewer import DicomSliceViewer

viewer = DicomSliceViewer(parent)
viewer.set_slice_count(volume.shape[0], initial_index=volume.shape[0] // 2)
viewer.set_hu_image(volume[viewer.current_slice])
viewer.slice_changed.connect(lambda index: viewer.set_hu_image(volume[index]))
```

Feature-specific controls can be appended with `add_toolbar_separator()` and
`add_toolbar_widget()`. Feature-specific overlays should subclass
`DicomImageView`, then pass that canvas through the `image_view` constructor
argument. This keeps segmentation hit testing, lesion markers, and similar
behavior out of the shared DICOM controls.

## Signals

- `DicomSliceViewer.slice_changed(int)` emits a zero-based slice index.
- `DicomSliceViewer.image_clicked(float, float)` emits image column and row.
- `DicomSliceViewer.window_level_changed(float, float)` emits width and level.
- `DicomImageView.slice_delta_requested(int)` exposes wheel navigation for a
  custom container.
- `DicomImageView.window_level_dragged(float, float)` exposes raw drag deltas for
  a custom windowing implementation.
