"""
Coordinate Mapping and Label Generation Module

Functions:
- map_coordinates_with_flip: BMP to NPY coordinate transformation
- create_pixel_mask: Generate binary pixel-wise mask
- save_labeled_data: Save NPZ format with intensity + labels + metadata
"""

# Import functions from verified modules
from .coordinate_mapper import CoordinateMapper
from .label_generator import LabelGenerator

__all__ = [
    'CoordinateMapper',
    'LabelGenerator'
]
