"""Pure-Python DICOM-CT-PD lesion insertion pipeline."""

from .config import InsertionConfig, InsertionSpec
from .pipeline import run_insertion

__all__ = ["InsertionConfig", "InsertionSpec", "run_insertion"]
__version__ = "3.1.4"
