from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QObject


@dataclass(slots=True)
class ModuleDescriptor:
    module_id: str
    display_name: str
    workspace: QWidget
    panel: QWidget | None
    status_text: str
    owner: QObject | None
    show_panel: bool = True
    is_busy: Callable[[], bool] = lambda: False
    request_cancel: Callable[[], None] = lambda: None
    activate: Callable[[], None] = lambda: None
    deactivate: Callable[[], None] = lambda: None
    cleanup: Callable[[], None] = lambda: None


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: list[ModuleDescriptor] = []
        self._by_id: dict[str, ModuleDescriptor] = {}

    def register(self, descriptor: ModuleDescriptor) -> None:
        if descriptor.module_id in self._by_id:
            raise ValueError(f"Module ID already registered: {descriptor.module_id}")
        self._modules.append(descriptor)
        self._by_id[descriptor.module_id] = descriptor

    def get(self, module_id: str) -> ModuleDescriptor:
        return self._by_id[module_id]

    def __iter__(self) -> Iterator[ModuleDescriptor]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)
