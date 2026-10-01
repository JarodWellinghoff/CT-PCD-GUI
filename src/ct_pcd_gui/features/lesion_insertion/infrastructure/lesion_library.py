from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import Event

import numpy as np

from ..domain.models import LesionLibraryItem, ValidationError

SUPPORTED_EXTENSIONS = frozenset({".npz", ".mat"})


def _loader():
    try:
        from .._vendor.lesion_pipeline.lesion_models import load_lesion_model
    except ImportError as exc:
        raise RuntimeError("The bundled lesion-model loader is unavailable.") from exc
    return load_lesion_model


class LocalLesionModelLibrary:
    """Local MAT/NPZ library backed by the coworker's strict model loader."""

    def scan(
        self,
        source: str | Path,
        cancel_event: Event | None = None,
    ) -> Iterable[LesionLibraryItem]:
        root = Path(source).expanduser()
        if root.is_file():
            yield self.inspect(root)
            return
        if not root.is_dir():
            raise NotADirectoryError(f"Lesion library does not exist: {root}")
        for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
            if cancel_event and cancel_event.is_set():
                return
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            item = self.inspect(path)
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = path.name
            yield LesionLibraryItem(
                path=item.path,
                lesion_id=item.lesion_id,
                display_name=item.display_name,
                relative_path=relative,
                compatible=item.compatible,
                channel_count=item.channel_count,
                dimensions_rcs=item.dimensions_rcs,
                spacing_rcs_mm=item.spacing_rcs_mm,
                metadata=item.metadata,
                warning=item.warning,
            )

    def inspect(self, source: str | Path) -> LesionLibraryItem:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Lesion model does not exist: {path}")
        try:
            model = _loader()(path)
            shape = tuple(int(value) for value in model.voi_hu.shape[:3])
            means = np.asarray(model.lesion_mean_hu_by_channel, dtype=float).tolist()
            backgrounds = np.asarray(model.old_background_hu, dtype=float).tolist()
            contrast = (np.asarray(means) - np.asarray(backgrounds)).tolist()
            metadata = {
                "lesion_number": int(model.lesion_number),
                "channels": int(model.channel_count),
                "dimensions_rcs": list(shape),
                "spacing_rcs_mm": [
                    float(model.row_spacing_mm),
                    float(model.column_spacing_mm),
                    float(model.slice_spacing_mm),
                ],
                "lesion_mean_hu_by_channel": means,
                "old_background_hu": backgrounds,
                "contrast_hu_by_channel": contrast,
                "old_background_method": str(model.old_background_method),
                "old_background_voxel_count": int(model.old_background_voxel_count),
                "kvp": float(model.kvp),
            }
            return LesionLibraryItem(
                path=str(path),
                lesion_id=f"Lesion{int(model.lesion_number):02d}",
                display_name=path.stem,
                relative_path=path.name,
                compatible=True,
                channel_count=int(model.channel_count),
                dimensions_rcs=shape,
                spacing_rcs_mm=(
                    float(model.row_spacing_mm),
                    float(model.column_spacing_mm),
                    float(model.slice_spacing_mm),
                ),
                metadata=metadata,
            )
        except Exception as exc:
            return LesionLibraryItem(
                path=str(path),
                lesion_id=path.stem,
                display_name=path.stem,
                relative_path=path.name,
                compatible=False,
                warning=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    def load_model(self, source: str | Path):
        item = self.inspect(source)
        if not item.compatible:
            raise ValidationError(item.warning or f"Incompatible lesion model: {source}")
        return _loader()(source)


def lesion_thumbnail(model: object, *, size: int = 128) -> np.ndarray:
    """Return a normalized maximum-intensity mask projection for display."""

    mask = np.asarray(getattr(model, "mask"), dtype=bool)
    if mask.ndim == 4:
        mask = mask[..., 0]
    if mask.ndim != 3 or not np.any(mask):
        return np.zeros((size, size), dtype=np.uint8)
    projection = np.max(mask, axis=2).astype(np.uint8) * 255
    try:
        from scipy.ndimage import zoom

        factors = (size / projection.shape[0], size / projection.shape[1])
        resized = zoom(projection, factors, order=0, prefilter=False)
        return resized[:size, :size].astype(np.uint8, copy=False)
    except Exception:
        return projection
