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

The preview supports series switching, slice scrolling, window/level controls,
overlay opacity, zooming with Ctrl+wheel, panning, and component selection.

## Multiple series

For one series, lesion files are written directly to the selected output folder.
For multiple series, the accepted components are resampled with nearest-neighbor
interpolation and written once per series in separate series subfolders. The NRRD
and DICOM data must share a valid physical coordinate system; the module does not
perform image registration.

## NPZ format

The export matches the reference `L005-1-Lesion.npz` field layout. `VOI` is a
signed 16-bit array and `LesionMask` is boolean; both use row/column/slice order
(`y, x, z`). DICOM ranges and `LesionCenter` use DICOM index order (`x, y, z`).
The archive contains no object arrays and loads with
`numpy.load(path, allow_pickle=False)`.

The crop uses the legacy half-lesion-size padding rule with a minimum one-voxel
margin. The file also includes DICOM header JSON, lesion geometry, HU statistics,
voxel count, and physical volume.

## Command line

```bash
ct-pcd-extract-lesions \
  --dicom /path/to/series \
  --segmentation /path/to/segments.seg.nrrd \
  --output /path/to/models \
  --segment Segment0 \
  --minimum-voxels 10
```

Repeat `--dicom`, `--segment`, or `--omit` as needed. Use `--list-labels` and
`--list-candidates` to inspect the segmentation without exporting.
