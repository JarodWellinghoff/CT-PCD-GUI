from pathlib import Path
from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QFormLayout,
    QLineEdit,
    QScrollArea,
)
import pydicom

# ===========================================================================
# DICOM Details dock
# ===========================================================================


class DicomDetailsDock(QDockWidget):
    DISPLAY_TAGS = [
        ("Patient Name", "PatientName"),
        ("Patient ID", "PatientID"),
        ("Patient Birth Date", "PatientBirthDate"),
        ("Patient Sex", "PatientSex"),
        ("Study Date", "StudyDate"),
        ("Study Description", "StudyDescription"),
        ("Study ID", "StudyID"),
        ("Accession Number", "AccessionNumber"),
        ("Series Date", "SeriesDate"),
        ("Series Description", "SeriesDescription"),
        ("Series Number", "SeriesNumber"),
        ("Modality", "Modality"),
        ("Manufacturer", "Manufacturer"),
        ("Manufacturer Model", "ManufacturerModelName"),
        ("Station Name", "StationName"),
        ("Institution Name", "InstitutionName"),
        ("Body Part", "BodyPartExamined"),
        ("Slice Thickness", "SliceThickness"),
        ("Spacing Between Slices", "SpacingBetweenSlices"),
        ("Pixel Spacing", "PixelSpacing"),
        ("KVP", "KVP"),
        ("Exposure (mAs)", "Exposure"),
        ("Convolution Kernel", "ConvolutionKernel"),
        ("Rows", "Rows"),
        ("Columns", "Columns"),
        ("Bits Allocated", "BitsAllocated"),
        ("Bits Stored", "BitsStored"),
        ("Window Center", "WindowCenter"),
        ("Window Width", "WindowWidth"),
        ("Rescale Intercept", "RescaleIntercept"),
        ("Rescale Slope", "RescaleSlope"),
    ]

    def __init__(self, parent=None):
        super().__init__("DICOM Details", parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        self.form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.fields: dict[str, QLineEdit] = {}
        for label, kw in self.DISPLAY_TAGS:
            f = QLineEdit()
            f.setReadOnly(True)
            f.setPlaceholderText("--")
            self.fields[kw] = f
            self.form_layout.addRow(f"{label}:", f)
        self.slices_field = QLineEdit()
        self.slices_field.setReadOnly(True)
        self.slices_field.setPlaceholderText("--")
        self.form_layout.addRow("Number of Slices:", self.slices_field)
        self.folder_field = QLineEdit()
        self.folder_field.setReadOnly(True)
        self.folder_field.setPlaceholderText("--")
        self.form_layout.addRow("Folder:", self.folder_field)
        scroll.setWidget(self.form_widget)
        self.setWidget(scroll)

    def clear(self):
        for f in self.fields.values():
            f.clear()
        self.slices_field.clear()
        self.folder_field.clear()

    def populate_from_folder(self, dicom_folder):
        self.clear()
        self.folder_field.setText(dicom_folder)
        folder = Path(dicom_folder)
        dcm_files = sorted(
            f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")
        )
        if not dcm_files:
            self.slices_field.setText("No files found")
            return
        self.slices_field.setText(str(len(dcm_files)))
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        except Exception as e:
            self.slices_field.setText(f"Error: {e}")
            return
        for _, kw in self.DISPLAY_TAGS:
            v = getattr(ds, kw, None)
            if v is not None:
                self.fields[kw].setText(str(v))
