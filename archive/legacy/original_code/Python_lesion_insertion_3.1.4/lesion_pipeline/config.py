from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InsertionSpec:
    """One model, placement, and new-location background values for reporting."""

    lesion_model: str
    background_hu: list[float]
    center_pixel: list[float] | None = None
    center_ctpd_mm: list[float] | None = None
    label: str = ""

    def validate(self) -> None:
        if not Path(self.lesion_model).is_file():
            raise FileNotFoundError(f"Lesion model not found: {self.lesion_model}")
        if (self.center_pixel is None) == (self.center_ctpd_mm is None):
            raise ValueError(
                "Each insertion must define exactly one of center_pixel or "
                "center_ctpd_mm."
            )
        center = self.center_pixel if self.center_pixel is not None else self.center_ctpd_mm
        if center is None or len(center) != 3:
            raise ValueError("An insertion center must contain exactly three values.")
        if not self.background_hu:
            raise ValueError("background_hu must contain at least one value.")


@dataclass
class InsertionConfig:
    """Serializable configuration for a complete DICOM-CT-PD insertion run."""

    ctpd_input: str
    output_dir: str
    insertions: list[InsertionSpec]
    t1_series_dir: str | None = None
    spectrum_channel_map: dict[int, int] = field(default_factory=dict)
    patient_position: str = "HFS"
    workers: int = 1
    overwrite: bool = False

    def validate(self) -> None:
        if not Path(self.ctpd_input).exists():
            raise FileNotFoundError(f"DICOM-CT-PD input not found: {self.ctpd_input}")
        if not self.insertions:
            raise ValueError("At least one insertion is required.")
        for insertion in self.insertions:
            insertion.validate()
            if insertion.center_pixel is not None and not self.t1_series_dir:
                raise ValueError(
                    "t1_series_dir is required when center_pixel is used."
                )
        if self.t1_series_dir and not Path(self.t1_series_dir).is_dir():
            raise NotADirectoryError(f"T1 series directory not found: {self.t1_series_dir}")
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")
        if self.patient_position.upper() not in {"HFS", "FFS"}:
            raise ValueError("patient_position must be HFS or FFS.")
        for spectrum, channel in self.spectrum_channel_map.items():
            if int(spectrum) < 1 or int(channel) < 0:
                raise ValueError(
                    "Spectrum indices are 1-based and lesion channels are 0-based."
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsertionConfig":
        insertions = [InsertionSpec(**item) for item in data.get("insertions", [])]
        spectrum_map = {
            int(key): int(value)
            for key, value in data.get("spectrum_channel_map", {}).items()
        }
        config = cls(
            ctpd_input=str(data["ctpd_input"]),
            output_dir=str(data["output_dir"]),
            insertions=insertions,
            t1_series_dir=data.get("t1_series_dir"),
            spectrum_channel_map=spectrum_map,
            patient_position=str(data.get("patient_position", "HFS")).upper(),
            workers=int(data.get("workers", 1)),
            overwrite=bool(data.get("overwrite", False)),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "InsertionConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)

        # Resolve relative paths against the JSON file, not the current directory.
        base = config_path.resolve().parent
        for key in ("ctpd_input", "output_dir", "t1_series_dir"):
            value = data.get(key)
            if value and not Path(value).is_absolute():
                data[key] = str(base / value)
        for item in data.get("insertions", []):
            value = item.get("lesion_model")
            if value and not Path(value).is_absolute():
                item["lesion_model"] = str(base / value)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["spectrum_channel_map"] = {
            str(key): value for key, value in self.spectrum_channel_map.items()
        }
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.to_dict(), stream, indent=2)
            stream.write("\n")
