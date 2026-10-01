from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..domain.models import LesionInstance, LesionSession, ValidationError


class LesionSessionState:
    """Mutable application state with immutable, independently editable lesions."""

    def __init__(self, session: LesionSession | None = None) -> None:
        self._session = session or LesionSession()
        self._selected_id = self._session.lesions[0].instance_id if self._session.lesions else ""
        self._saved_snapshot = self._session
        self._revision = 0

    @property
    def session(self) -> LesionSession:
        return self._session

    @property
    def selected_id(self) -> str:
        return self._selected_id

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def is_dirty(self) -> bool:
        return self._session != self._saved_snapshot

    @property
    def selected(self) -> LesionInstance | None:
        return next(
            (item for item in self._session.lesions if item.instance_id == self._selected_id),
            None,
        )

    def snapshot(self) -> tuple[LesionSession, str]:
        return self._session, self._selected_id

    def restore(self, snapshot: tuple[LesionSession, str]) -> None:
        session, selected_id = snapshot
        session.validate_basic()
        self._session = session
        self._selected_id = selected_id if any(
            item.instance_id == selected_id for item in session.lesions
        ) else (session.lesions[0].instance_id if session.lesions else "")
        self._revision += 1

    def replace_session(self, session: LesionSession, *, mark_saved: bool = False) -> None:
        session.validate_basic()
        self._session = session
        self._selected_id = session.lesions[0].instance_id if session.lesions else ""
        if mark_saved:
            self._saved_snapshot = session
        self._revision += 1

    def update_session(self, **changes) -> None:
        self._session = replace(self._session, **changes)
        self._session.validate_basic()
        self._revision += 1

    def select(self, instance_id: str) -> None:
        if instance_id and not any(
            lesion.instance_id == instance_id for lesion in self._session.lesions
        ):
            raise KeyError(instance_id)
        self._selected_id = instance_id

    def add(self, lesion: LesionInstance, *, index: int | None = None) -> None:
        if any(item.instance_id == lesion.instance_id for item in self._session.lesions):
            raise ValidationError(f"Lesion instance already exists: {lesion.instance_id}")
        values = list(self._session.lesions)
        if index is None:
            values.append(lesion)
        else:
            values.insert(max(0, min(index, len(values))), lesion)
        self._session = replace(self._session, lesions=tuple(values))
        self._selected_id = lesion.instance_id
        self._revision += 1

    def update(self, lesion: LesionInstance) -> None:
        values = list(self._session.lesions)
        for index, current in enumerate(values):
            if current.instance_id == lesion.instance_id:
                values[index] = lesion
                self._session = replace(self._session, lesions=tuple(values))
                self._selected_id = lesion.instance_id
                self._revision += 1
                return
        raise KeyError(lesion.instance_id)

    def remove(self, instance_id: str) -> LesionInstance:
        values = list(self._session.lesions)
        for index, current in enumerate(values):
            if current.instance_id == instance_id:
                removed = values.pop(index)
                self._session = replace(self._session, lesions=tuple(values))
                if self._selected_id == instance_id:
                    self._selected_id = (
                        values[min(index, len(values) - 1)].instance_id if values else ""
                    )
                self._revision += 1
                return removed
        raise KeyError(instance_id)

    def duplicate(self, instance_id: str) -> LesionInstance:
        values = list(self._session.lesions)
        for index, current in enumerate(values):
            if current.instance_id == instance_id:
                copy = current.duplicate()
                values.insert(index + 1, copy)
                self._session = replace(self._session, lesions=tuple(values))
                self._selected_id = copy.instance_id
                self._revision += 1
                return copy
        raise KeyError(instance_id)

    def move(self, instance_id: str, offset: int) -> None:
        values = list(self._session.lesions)
        source = next(
            (index for index, item in enumerate(values) if item.instance_id == instance_id),
            -1,
        )
        if source < 0:
            raise KeyError(instance_id)
        destination = max(0, min(source + offset, len(values) - 1))
        if destination == source:
            return
        item = values.pop(source)
        values.insert(destination, item)
        self._session = replace(self._session, lesions=tuple(values))
        self._revision += 1

    def set_all_visibility(self, visible: bool) -> None:
        self._session = replace(
            self._session,
            lesions=tuple(replace(item, visible=visible) for item in self._session.lesions),
        )
        self._revision += 1

    def mark_saved(self) -> None:
        self._saved_snapshot = self._session

    def save(self, path: str | Path) -> Path:
        output = self._session.save(path)
        self.mark_saved()
        return output

    def load(self, path: str | Path) -> LesionSession:
        session = LesionSession.load(path)
        self.replace_session(session, mark_saved=True)
        return session
