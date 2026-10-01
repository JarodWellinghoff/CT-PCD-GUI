# CT-PCD-GUI

**Version:** 0.1.0  
**Author:** Jarod Wellinghoff

PySide6 research workbench for DICOM-CT-PD processing, lesion extraction,
lesion-model viewing, interactive lesion insertion, and noise insertion.

## Run

```bash
python -m pip install -e ".[dev,lesion-extraction]"
ct-pcd-gui
```

The `lesion-extraction` extra installs SimpleITK for DICOM/NRRD geometry,
resampling, connected-component analysis, and lesion statistics.

## Development

```bash
pytest
ruff check .
pyright
```

See [Reusable reconstructed-DICOM viewer](docs/dicom_viewer.md) for the shared
slice controls, mouse interactions, signals, and extension points used by GUI
modules that preview reconstructed DICOM images.

See [Lesion extraction module](docs/lesion_extraction.md) for the NRRD-to-NPZ
workflow, connected-component review, multi-series behavior, file schema, and
CLI usage.

See [Lesion insertion module](docs/lesion_insertion.md) for the workflow,
coordinate conventions, output manifest, external reconstruction integration,
limitations, and manual validation checklist.
