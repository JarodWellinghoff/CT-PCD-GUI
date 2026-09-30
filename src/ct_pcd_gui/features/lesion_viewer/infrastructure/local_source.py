from collections.abc import Iterable
import os
from pathlib import Path
import threading
from typing import Callable

from ..domain.models import LesionFileRecord

SUPPORTED_EXTENSIONS = frozenset({".mat", ".npz"})


class LocalDirectoryLesionSource:
    """Recursively enumerates local MAT/NPZ lesion files."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
        extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.include_hidden = include_hidden
        self.follow_symlinks = follow_symlinks
        self.extensions = frozenset(ext.lower() for ext in extensions)

    @property
    def display_name(self) -> str:
        return str(self.root)

    def iter_records(
        self,
        cancel_event: threading.Event | None = None,
    ) -> Iterable[LesionFileRecord]:
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Lesion library directory does not exist: {self.root}"
            )

        root_str = str(self.root)
        for current_root, dirnames, filenames in os.walk(
            root_str,
            followlinks=self.follow_symlinks,
        ):
            if cancel_event and cancel_event.is_set():
                return

            if not self.include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            current = Path(current_root)
            for filename in filenames:
                if cancel_event and cancel_event.is_set():
                    return
                if not self.include_hidden and filename.startswith("."):
                    continue

                path = current / filename
                if path.suffix.lower() not in self.extensions:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                try:
                    relative = str(path.relative_to(self.root))
                except ValueError:
                    relative = path.name
                yield LesionFileRecord(
                    path=path,
                    relative_path=relative,
                    size_bytes=int(stat.st_size),
                    modified_time=float(stat.st_mtime),
                )

    def scan(
        self,
        cancel_event: threading.Event | None = None,
        batch_callback: Callable[[list[LesionFileRecord]], None] | None = None,
        *,
        batch_size: int = 100,
    ) -> list[LesionFileRecord]:
        records: list[LesionFileRecord] = []
        batch: list[LesionFileRecord] = []
        for record in self.iter_records(cancel_event):
            records.append(record)
            if batch_callback is not None:
                batch.append(record)
                if len(batch) >= batch_size:
                    batch_callback(batch)
                    batch = []
        if batch_callback is not None and batch:
            batch_callback(batch)
        records.sort(key=lambda item: item.relative_path.casefold())
        return records
