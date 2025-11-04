"""
Mine Labeling Core Package

Modules:
- extractors: XTF data extraction
- labeling: Coordinate mapping and label generation
- sampling: Mine and background sampling
- augmentation: Data augmentation techniques
- validation: Data validation and verification
- utils: Utility functions
"""

from .extractors import *
from .labeling import *
from .sampling import *
from .augmentation import *
from .validation import *
from .utils import *

__all__ = [
    'extractors',
    'labeling',
    'sampling',
    'augmentation',
    'validation',
    'utils'
]
