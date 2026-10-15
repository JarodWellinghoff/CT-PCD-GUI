"""Per-placement source threshold and target threshold controls."""
from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel

from ..domain.thresholds import threshold_summary


class ThresholdSelector(QGroupBox):
    selection_changed = Signal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__("Thresholds for this lesion", parent)
        self._lesion = None
        self._spectra: tuple[int, ...] = ()
        layout = QFormLayout(self)
        self.source_combo = QComboBox()
        self.source_combo.setToolTip(
            "Select the source data in the lesion file. Source numbers identify "
            "file channels, not an HU cutoff. They are not automatically matched "
            "to target scan threshold numbers."
        )
        self.target_combo = QComboBox()
        self.target_combo.setToolTip(
            "Choose which raw-scan spectrum IDs receive this lesion. The other "
            "thresholds receive no contribution from this lesion."
        )
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.note = QLabel(
            "Choose the source first, then its destination(s). These choices "
            "apply only to the selected placed lesion. The image-domain preview "
            "shows source-threshold placement, not a target-specific reconstruction."
        )
        self.note.setWordWrap(True)
        layout.addRow("1. Lesion source threshold", self.source_combo)
        layout.addRow("2. Insert into target scan", self.target_combo)
        layout.addRow(self.summary)
        layout.addRow(self.note)
        self.source_combo.currentIndexChanged.connect(self._changed)
        self.target_combo.currentIndexChanged.connect(self._changed)
        self.set_lesion(None)

    def set_available_spectra(self, spectra) -> None:
        values = tuple(sorted(set(spectra)))
        if values != self._spectra:
            self._spectra = values
            self.set_lesion(self._lesion)

    def set_lesion(self, lesion) -> None:
        self._lesion = lesion
        with QSignalBlocker(self.source_combo), QSignalBlocker(self.target_combo):
            self.source_combo.clear()
            self.target_combo.clear()
            self.source_combo.addItem("Choose a lesion threshold...", None)
            self.target_combo.addItem(
                "Choose destination(s)..." if self._spectra else "Inspect raw CTPD first",
                (),
            )
            if lesion is not None:
                count = len(lesion.background_hu)
                for channel in range(count):
                    suffix = " (only source)" if count == 1 else ""
                    self.source_combo.addItem(f"Lesion threshold {channel + 1}{suffix}", channel)
                source_index = self.source_combo.findData(lesion.source_channel)
                if source_index < 0 and lesion.source_channel is not None:
                    self.source_combo.addItem(
                        "Unavailable source threshold - choose again", lesion.source_channel
                    )
                    source_index = self.source_combo.count() - 1
                self.source_combo.setCurrentIndex(max(0, source_index))
                if len(self._spectra) > 1:
                    prefix = "Both" if len(self._spectra) == 2 else "All"
                    self.target_combo.addItem(
                        f"{prefix} target thresholds ({', '.join(map(str, self._spectra))})",
                        self._spectra,
                    )
                for spectrum in self._spectra:
                    self.target_combo.addItem(f"Target threshold {spectrum} only", (spectrum,))
                targets = tuple(sorted(lesion.target_spectrum_indices))
                target_index = next((
                    index for index in range(self.target_combo.count())
                    if tuple(self.target_combo.itemData(index) or ()) == targets
                ), -1)
                if target_index < 0 and targets:
                    self.target_combo.addItem(
                        "Saved targets (review): " + ", ".join(map(str, targets)), targets
                    )
                    target_index = self.target_combo.count() - 1
                self.target_combo.setCurrentIndex(max(0, target_index))
            self.setEnabled(lesion is not None)
            self.target_combo.setEnabled(lesion is not None and bool(self._spectra))
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        if self._lesion is None:
            self.summary.setText("Add or select a placed lesion to choose its thresholds.")
            return
        self.summary.setText(threshold_summary(
            self.source_combo.currentData(),
            tuple(self.target_combo.currentData() or ()),
            self._spectra if self._spectra else None,
        ))

    def _changed(self, _index: int) -> None:
        if self._lesion is None:
            return
        self._refresh_summary()
        self.selection_changed.emit(
            self.source_combo.currentData(), tuple(self.target_combo.currentData() or ())
        )
