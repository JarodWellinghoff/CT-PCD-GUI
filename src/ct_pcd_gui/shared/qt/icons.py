from PySide6.QtWidgets import QStyle, QWidget


def standard_icon(widget: QWidget, pixmap_name: str):
    return widget.style().standardIcon(getattr(QStyle.StandardPixmap, pixmap_name))
