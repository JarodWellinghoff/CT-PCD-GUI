from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    """Small dependency-free approximation of ctkCollapsibleButton."""

    def __init__(
        self, title: str, expanded: bool = True, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.toggle.setStyleSheet(
            "QToolButton { background: #4a4a4a; border: 1px solid #5a5a5a; "
            "border-radius: 2px; padding: 5px; text-align: left; font-weight: bold; }"
            "QToolButton:hover { background: #545454; }"
        )

        self.content = QWidget(self)
        self.content.setVisible(expanded)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 7, 5, 7)
        self.content_layout.setSpacing(7)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

        self.toggle.toggled.connect(self.set_expanded)

    @Slot(bool)
    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(expanded)

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.content_layout.addLayout(layout)
