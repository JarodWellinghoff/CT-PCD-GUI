# Lesion extraction module

The **Lesion Extraction** module converts a labeled NRRD segmentation and one or
more CT DICOM series into lesion-model `.npz` files accepted by the lesion viewer
and lesion insertion workflows.

## Workflow

1. Add one or more folders containing DICOM image series.
2. Select a `.nrrd` or `.seg.nrrd` segmentation and choose **Load and preview**.
3. Review the segmentation overlay and check the segment labels that should be
   treated as lesions. Labels containing “lesion” are selected initially; if no
   label has that word, all non-zero labels are selected.
4. Set an optional minimum component size and choose **Preview lesion split**.
   Each selected label is split into fully connected 3-D components.
5. Uncheck components that should be omitted and edit output names as needed.
6. Select an output folder and export.

The preview uses the shared reconstructed-DICOM viewer controls: Fit/Reset,
window presets, mouse-wheel slice navigation, Ctrl+wheel zoom, middle-button
panning, and right-drag window/level. Segmentation visibility and opacity remain
available beside the shared controls, and overlays can still be clicked to select
components.

## Multiple series

The module writes exactly one `.npz` file for each included lesion, regardless of
the number of input series. Input series order defines channel order. For multiple
series, `VOI` and `LesionMask` use shape `(row, column, slice, series)` and each
channel contains the data from the corresponding DICOM series.

The first input series is the reference series for lesion geometry, ranges,
spacing, and the legacy scalar DICOM header fields. Additional arrays record the
ordered series names, UIDs, source paths, DICOM headers, and channel-specific HU
statistics.

All input series must already be voxel-for-voxel aligned: size, spacing, origin,
and direction must match within numerical tolerance. The exporter validates this
before writing any lesion files. It does not register or interpolate DICOM series.
The NRRD segmentation is resampled to the reference series with nearest-neighbor
interpolation.

## NPZ format

For one input series, the export retains the exact 22-field layout of the
reference `L005-1-Lesion.npz`. `VOI` is a signed 16-bit array and `LesionMask` is
boolean; both use row/column/slice order (`y, x, z`). DICOM ranges and
`LesionCenter` use DICOM index order (`x, y, z`).

For multiple series, the same 22 fields remain first and the following fields are
appended:

- `SeriesCount`
- `ReferenceSeriesIndex`
- `SeriesNames`
- `SeriesUIDs`
- `SeriesSourcePaths`
- `DicomHeadersJSON`
- `LesionMeanHUByChannel`
- `LesionMaxHUByChannel`
- `LesionMinHUByChannel`
- `LesionMedianHUByChannel`
- `LesionSigmaByChannel`
- `LesionVarianceByChannel`

The archive contains no object arrays and loads with
`numpy.load(path, allow_pickle=False)`. The crop uses the legacy half-lesion-size
padding rule with a minimum one-voxel margin. Geometry, voxel count, and physical
volume are shared across channels because aligned series represent the same
lesion voxels.

## Command line

```bash
ct-pcd-extract-lesions \
  --dicom /path/to/series-a \
  --dicom /path/to/series-b \
  --segmentation /path/to/segments.seg.nrrd \
  --output /path/to/models \
  --segment Segment0 \
  --minimum-voxels 10
```

Repeat `--dicom`, `--segment`, or `--omit` as needed. Use `--list-labels` and
`--list-candidates` to inspect the segmentation without exporting.
