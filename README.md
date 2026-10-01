# CT-PCD-GUI

**Version:** 0.1.0  
**Author:** Jarod Wellinghoff

PySide6 research workbench for DICOM-CT-PD processing, lesion-model viewing, and interactive lesion insertion.

## Run

```bash
python -m pip install -e ".[dev]"
ct-pcd-gui
```

## Development

```bash
pytest
ruff check .
pyright
```

See [Lesion insertion module](docs/lesion_insertion.md) for the workflow, coordinate conventions, output manifest, external reconstruction integration, limitations, and manual validation checklist.
