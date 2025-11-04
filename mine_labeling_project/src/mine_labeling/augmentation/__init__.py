"""
Data Augmentation Module

Classes:
- Augmentor: Apply 9 augmentation techniques
- AugmentationPipeline: Manage augmentation workflow

Techniques:
1. Horizontal flip
2. Vertical flip
3. Rotation (±15°)
4. Scale (±10%)
5. Brightness (±20%)
6. Contrast (±20%)
7. Gaussian blur
8. Gaussian noise
9. Elastic transform
"""

from .augmentor import Augmentor

__all__ = [
    'Augmentor'
]
