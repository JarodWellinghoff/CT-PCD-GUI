from pathlib import Path
import json


SETTINGS_FILE = Path.home() / ".dicom_viewer_settings.json"


# ===========================================================================
# Settings persistence
# ===========================================================================


class SettingsManager:
    """JSON-based persistence for application state."""

    DEFAULTS = {
        "lesion_library_dir": "",
        "last_dicom_folder": "",
        "custom_spacing": [1.0, 1.0, 1.0],
        "use_dicom_spacing": True,
        "window_geometry": [100, 100, 1200, 800],
        "rois": [],  # list of serialised ROI dicts
    }

    def __init__(self, path: Path = SETTINGS_FILE):
        self._path = path
        self._data: dict = dict(self.DEFAULTS)
        self.load()

    # ---- I/O ----

    def load(self):
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    stored = json.load(f)
                for k, v in self.DEFAULTS.items():
                    self._data[k] = stored.get(k, v)
            except Exception:
                self._data = dict(self.DEFAULTS)

    def save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save settings: {e}")

    # ---- Accessors ----

    def get(self, key: str):
        return self._data.get(key, self.DEFAULTS.get(key))

    def set(self, key: str, value):
        self._data[key] = value

    # ---- ROI serialisation helpers ----

    @staticmethod
    def serialise_rois(roi_dict: dict) -> list:
        out = []
        for roi in roi_dict.values():
            out.append(
                {
                    "name": roi.name,
                    "roi_type": roi.roi_type,
                    "slice_index": roi.slice_index,
                    "orientation": roi.orientation,
                    "points": [list(p) for p in roi.points],
                    "color": list(roi.color),
                }
            )
        return out

    @staticmethod
    def deserialise_rois(data: list) -> list:
        """Return list of dicts ready for ROIManager.add_roi()."""
        out = []
        for d in data:
            out.append(
                {
                    "name": d["name"],
                    "roi_type": d["roi_type"],
                    "slice_index": d["slice_index"],
                    "orientation": d["orientation"],
                    "points": [tuple(p) for p in d["points"]],
                    "color": tuple(d["color"]),
                }
            )
        return out
