from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
)

from ct_pcd_gui.features.lesion_viewer.domain.models import LesionFileRecord


class LesionFileTableModel(QAbstractTableModel):
    COLUMNS = ("Name", "Folder", "Type", "Size")
    RECORD_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[LesionFileRecord] = []

    @property
    def records(self) -> tuple[LesionFileRecord, ...]:
        return tuple(self._records)

    def clear(self) -> None:
        self.beginResetModel()
        self._records.clear()
        self.endResetModel()

    def replace(self, records: Sequence[LesionFileRecord]) -> None:
        self.beginResetModel()
        self._records = list(records)
        self.endResetModel()

    def append_batch(self, records: Sequence[LesionFileRecord]) -> None:
        if not records:
            return
        first = len(self._records)
        last = first + len(records) - 1
        self.beginInsertRows(QModelIndex(), first, last)
        self._records.extend(records)
        self.endInsertRows()

    def record_at(self, row: int) -> LesionFileRecord | None:
        if 0 <= row < len(self._records):
            return self._records[row]
        return None

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        record = self.record_at(index.row())
        if record is None:
            return None

        if role == self.RECORD_ROLE:
            return record
        if role == Qt.ItemDataRole.ToolTipRole:
            return (
                f"{record.path}\n"
                f"Relative path: {record.relative_path}\n"
                f"Size: {record.size_bytes:,} bytes"
            )
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() == 3:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if index.column() == 0:
            return record.name
        if index.column() == 1:
            parent = str(Path(record.relative_path).parent)
            return "" if parent == "." else parent
        if index.column() == 2:
            return record.suffix.upper().lstrip(".")
        if index.column() == 3:
            return _format_size(record.size_bytes)
        return None

    def headerData(  # noqa: N802 - Qt API
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
            and 0 <= section < len(self.COLUMNS)
        ):
            return self.COLUMNS[section]
        return None

    def sort(  # noqa: A003 - Qt method name
        self,
        column: int,
        order=Qt.SortOrder.AscendingOrder,
    ) -> None:
        reverse = order == Qt.SortOrder.DescendingOrder

        def key(record: LesionFileRecord):
            if column == 0:
                return record.name.casefold()
            if column == 1:
                return record.relative_path.casefold()
            if column == 2:
                return record.suffix.casefold()
            if column == 3:
                return record.size_bytes
            return record.relative_path.casefold()

        self.layoutAboutToBeChanged.emit()
        self._records.sort(key=key, reverse=reverse)
        self.layoutChanged.emit()


class LesionFileFilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._query = ""
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_query(self, text: str) -> None:
        query = text.strip().casefold()
        if query == self._query:
            return
        self._query = query
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # noqa: N802
        if not self._query:
            return True
        model = self.sourceModel()
        if not isinstance(model, LesionFileTableModel):
            return True
        record = model.record_at(source_row)
        if record is None:
            return False
        haystack = f"{record.name}\n{record.relative_path}".casefold()
        return all(token in haystack for token in self._query.split())


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes:,} B"
