from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

VIEW_COLORS = {
    "Red": "#f34a33",
    "Yellow": "#edd54c",
    "Green": "#6eb04b",
    "3D": "#7e7ecf",
}


class ViewPane(QFrame):
    """A colored header bar over a render area. Put VTK/pyqtgraph in .body."""

    def __init__(self, name, color, letter, parent=None):
        super().__init__(parent)
        self.name = name
        self.color = color

        self.header = QFrame(self)
        self.header.setFixedHeight(22)
        self.header.setStyleSheet(
            f"QFrame {{ background: {color}; }}"
            f"QLabel {{ color: #101010; font-weight: bold; }}"
            "QToolButton { border: 0; padding: 1px; }"
            "QToolButton:hover { background: rgba(0,0,0,60); }"
        )
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(4, 0, 6, 0)
        self.header_layout.setSpacing(4)

        self.pin = QToolButton(self.header)
        self.pin.setText("\u25c2")  # expands the controller row
        self.pin.setCheckable(True)
        self.pin.setToolTip("Show view controls")
        self.header_layout.addWidget(self.pin)
        self.header_layout.addWidget(QLabel(letter, self.header))

        self.maximize = QToolButton(self.header)
        self.maximize.setText("\u2b1c")
        self.maximize.setToolTip("Maximize view")
        self.maximize.setCheckable(True)

        # Controller row (hidden until the pin is toggled), Slicer's second bar.
        self.controller = QWidget(self)
        self.controller.setVisible(False)
        crow = QHBoxLayout(self.controller)
        crow.setContentsMargins(4, 2, 4, 2)
        crow.setSpacing(4)
        self.controller_layout = crow
        self.pin.toggled.connect(self.controller.setVisible)
        self.pin.toggled.connect(
            lambda on: self.pin.setText("\u25be" if on else "\u25c2")
        )

        self.body = QWidget(self)
        self.body.setStyleSheet("background: #000000;")
        self.body.setMinimumSize(QSize(80, 60))
        self.body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(self.controller)
        lay.addWidget(self.body, 1)

    def set_view_widget(self, w):
        """Swap the black placeholder for a real renderer."""
        layout = self.layout()
        if layout is not None:
            layout.replaceWidget(self.body, w)
        self.body.deleteLater()
        self.body = w


class SliceViewPane(ViewPane):
    """Adds the slice-offset slider and a volume selector, like Slicer."""

    offsetChanged = Signal(int)

    def __init__(self, name, color, letter, axis="S", parent=None):
        super().__init__(name, color, letter, parent)

        self.offset = QSlider(Qt.Orientation.Horizontal, self.header)
        self.offset.setRange(-200, 200)
        self.offset.setStyleSheet(
            "QSlider::groove:horizontal { height: 3px; background: rgba(0,0,0,90); }"
            "QSlider::handle:horizontal { background: #202020; width: 9px;"
            " margin: -5px 0; border-radius: 2px; }"
        )
        self.offset_label = QLabel(f"{axis}: 0.00mm", self.header)
        self.offset_label.setMinimumWidth(76)
        self.offset.valueChanged.connect(
            lambda v: self.offset_label.setText(f"{axis}: {v / 10:.2f}mm")
        )
        self.offset.valueChanged.connect(self.offsetChanged)

        self.header_layout.insertWidget(2, self.maximize)
        self.header_layout.addWidget(self.offset, 1)
        self.header_layout.addWidget(self.offset_label)

        self.orientation = QComboBox(self.controller)
        self.orientation.addItems(["Axial", "Sagittal", "Coronal", "Reformat"])
        self.volume = QComboBox(self.controller)
        self.volume.addItem("None")
        self.volume.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.controller_layout.addWidget(self.orientation)
        self.controller_layout.addWidget(self.volume, 1)


class ThreeDViewPane(ViewPane):
    def __init__(self, parent=None):
        super().__init__("3D", VIEW_COLORS["3D"], "1", parent)
        self.header_layout.addWidget(self.maximize)
        self.header_layout.addStretch(1)
        self.body.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            " stop:0 #6a6a9e, stop:1 #1b1b2b);"
        )
