from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


from ct_pcd_gui.shared.qt.collapsible_section import CollapsibleSection
from ct_pcd_gui.shared.qt.icons import standard_icon


class WelcomePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        title = QLabel("DICOM CTPD Workbench")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        title.setContentsMargins(6, 6, 6, 6)

        # Scrolling stack of collapsible sections.
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 0, 4, 4)
        inner_layout.setSpacing(4)

        load = CollapsibleSection("Load data", expanded=True)
        for label, pixmap in [
            ("Add Data", "SP_DialogOpenButton"),
            ("Add DICOM Data", "SP_DirIcon"),
            ("Download Sample Data", "SP_ArrowDown"),
            ("Install Extensions", "SP_FileDialogNewFolder"),
        ]:
            b = QToolButton()
            b.setText(label)
            b.setIcon(standard_icon(self, pixmap))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setStyleSheet(
                "QToolButton { background: #454545; border: 1px solid"
                " #5a5a5a; border-radius: 2px; padding: 5px;"
                " text-align: left; }"
                "QToolButton:hover { background: #505050; }"
            )
            load.add_widget(b)
        inner_layout.addWidget(load)

        for name in [
            "Feedback",
            "About",
            "Documentation && Tutorials",
            "Updates",
            "Acknowledgment",
        ]:
            sec = CollapsibleSection(name)
            body = QLabel(f"{name} content goes here.")
            body.setWordWrap(True)
            sec.add_widget(body)
            inner_layout.addWidget(sec)
        inner_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        # Data Probe pinned to the bottom, as in Slicer.
        probe = CollapsibleSection("Data Probe", expanded=True)
        self.probe_label = QLabel("L\nF\nB")
        self.probe_label.setStyleSheet("font-family: monospace;")
        probe.add_widget(self.probe_label)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(title)
        lay.addWidget(scroll, 1)
        lay.addWidget(probe)
