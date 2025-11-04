"""
Data Validation Module

Functions:
- validate_dimensions: Check array dimensions
- validate_scaling: Verify coordinate scaling factors
- validate_labels: Check label values and coverage
- validate_coordinates: Verify bounding box bounds
- generate_visualizations: Create validation visualizations
"""

from .validator import MappingValidator

__all__ = [
    'MappingValidator'
]
