"""
ai-image-db

Modules:
  - metadata_extractor / loader : extract ComfyUI/A1111/EXIF/C2PA and store in SQLite
  - captioner                   : Florence-2 + WD14-style tagging
  - provenance                  : C2PA + watermark / signature inspection (EU AI Act Art. 50 oriented)
"""

from .database import Database
from .loader import ImageLoader
from .metadata_extractor import MetadataExtractor
from .captioner import Captioner
from .provenance import ProvenanceChecker

__version__ = "2.0.0"
__all__ = [
    "Database",
    "ImageLoader",
    "MetadataExtractor",
    "Captioner",
    "ProvenanceChecker",
]
