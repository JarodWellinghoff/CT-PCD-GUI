"""Pure-Python DICOM-CT-PD lesion insertion pipeline."""

from . import lesion_models as _lesion_models
from . import pipeline as _pipeline
from .config import InsertionConfig, InsertionSpec
from .lesion_model_compat import load_lesion_model

_lesion_models.load_lesion_model = load_lesion_model
_pipeline.load_lesion_model = load_lesion_model
run_insertion = _pipeline.run_insertion

__all__ = ["InsertionConfig", "InsertionSpec", "load_lesion_model", "run_insertion"]
__version__ = "3.1.4"
