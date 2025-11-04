"""
Coordinate Mapping Module

Maps bounding box coordinates from BMP annotations to NPY intensity data
with configurable Y-axis flip support.

Verified: 2025-11-04
Based on: scripts/mine_labeling/06_flip_bbox_mapping.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class CoordinateMapper:
    """
    Maps coordinates from BMP annotation space to NPY intensity data space

    Handles:
    - X-axis scaling (1024 -> 6400, factor: 6.25)
    - Y-axis flip (optional, for existing BMP data)
    - Coordinate validation and bounds checking
    """

    def __init__(self,
                 bmp_width: int = 1024,
                 bmp_height: int = 5137,
                 npy_width: int = 6400,
                 npy_height: int = 5137,
                 apply_y_flip: bool = True):
        """
        Initialize coordinate mapper

        Args:
            bmp_width: BMP image width (default: 1024)
            bmp_height: BMP image height (default: 5137)
            npy_width: NPY array width (default: 6400)
            npy_height: NPY array height (default: 5137)
            apply_y_flip: Apply Y-axis flip (default: True for existing BMP data)
        """
        self.bmp_width = bmp_width
        self.bmp_height = bmp_height
        self.npy_width = npy_width
        self.npy_height = npy_height
        self.apply_y_flip = apply_y_flip

        # Calculate scaling factors
        self.scale_x = npy_width / bmp_width
        self.scale_y = npy_height / bmp_height

    def map_bbox(self, bbox: Dict[str, int]) -> Dict[str, int]:
        """
        Map single bounding box from BMP to NPY coordinates

        Args:
            bbox: Dictionary with keys: xmin, ymin, xmax, ymax

        Returns:
            Mapped bounding box in NPY coordinates
        """
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']

        # Apply Y-axis flip if enabled
        if self.apply_y_flip:
            ymin_flipped = (self.bmp_height - 1) - ymax
            ymax_flipped = (self.bmp_height - 1) - ymin

            ymin = ymin_flipped
            ymax = ymax_flipped

        # Scale coordinates
        mapped = {
            'xmin': int(xmin * self.scale_x),
            'ymin': int(ymin * self.scale_y),
            'xmax': int(xmax * self.scale_x),
            'ymax': int(ymax * self.scale_y)
        }

        # Calculate dimensions
        mapped['width'] = mapped['xmax'] - mapped['xmin']
        mapped['height'] = mapped['ymax'] - mapped['ymin']

        # Calculate center
        mapped['center_x'] = (mapped['xmin'] + mapped['xmax']) // 2
        mapped['center_y'] = (mapped['ymin'] + mapped['ymax']) // 2

        return mapped

    def map_point(self, x: float, y: float) -> Tuple[int, int]:
        """
        Map single point from BMP to NPY coordinates

        Args:
            x: X coordinate in BMP space
            y: Y coordinate in BMP space

        Returns:
            Tuple (x_npy, y_npy)
        """
        # Apply Y-axis flip if enabled
        if self.apply_y_flip:
            y = (self.bmp_height - 1) - y

        # Scale coordinates
        x_npy = int(x * self.scale_x)
        y_npy = int(y * self.scale_y)

        return x_npy, y_npy

    def map_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map multiple annotations from BMP to NPY coordinates

        Args:
            annotations: List of annotation dictionaries with bounding boxes

        Returns:
            List of mapped annotations with original and mapped coordinates
        """
        mapped_annotations = []

        for ann in annotations:
            # Extract original bbox
            original_bbox = {
                'xmin': ann['xmin'],
                'ymin': ann['ymin'],
                'xmax': ann['xmax'],
                'ymax': ann['ymax']
            }

            # Map to NPY coordinates
            mapped_bbox = self.map_bbox(original_bbox)

            # Build result
            mapped_ann = {
                'name': ann.get('name', 'object'),
                'original_bmp': original_bbox.copy(),
                'mapped_npy': mapped_bbox
            }

            # Add center point if available
            if 'center_x' in ann and 'center_y' in ann:
                center_x, center_y = self.map_point(ann['center_x'], ann['center_y'])
                mapped_ann['center_npy'] = {
                    'x': center_x,
                    'y': center_y
                }

            mapped_annotations.append(mapped_ann)

        return mapped_annotations

    def validate_bbox(self, bbox: Dict[str, int]) -> bool:
        """
        Validate bounding box is within NPY bounds

        Args:
            bbox: Mapped bounding box

        Returns:
            True if valid, False otherwise
        """
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']

        # Check bounds
        if xmin < 0 or xmax > self.npy_width:
            return False

        if ymin < 0 or ymax > self.npy_height:
            return False

        # Check consistency
        if xmin >= xmax or ymin >= ymax:
            return False

        return True

    def clip_bbox(self, bbox: Dict[str, int]) -> Dict[str, int]:
        """
        Clip bounding box to NPY bounds

        Args:
            bbox: Bounding box to clip

        Returns:
            Clipped bounding box
        """
        clipped = {
            'xmin': max(0, min(bbox['xmin'], self.npy_width - 1)),
            'ymin': max(0, min(bbox['ymin'], self.npy_height - 1)),
            'xmax': max(0, min(bbox['xmax'], self.npy_width)),
            'ymax': max(0, min(bbox['ymax'], self.npy_height))
        }

        # Recalculate dimensions
        clipped['width'] = clipped['xmax'] - clipped['xmin']
        clipped['height'] = clipped['ymax'] - clipped['ymin']

        return clipped

    def get_mapping_info(self) -> Dict[str, Any]:
        """
        Get mapping configuration information

        Returns:
            Dictionary with mapping parameters
        """
        return {
            'bmp_dimensions': {
                'width': self.bmp_width,
                'height': self.bmp_height
            },
            'npy_dimensions': {
                'width': self.npy_width,
                'height': self.npy_height
            },
            'scaling_factors': {
                'scale_x': self.scale_x,
                'scale_y': self.scale_y
            },
            'transformations': {
                'apply_y_flip': self.apply_y_flip,
                'flip_formula': 'Y_npy = (bmp_height - 1) - Y_bmp' if self.apply_y_flip else 'None'
            }
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CoordinateMapper':
        """
        Create CoordinateMapper from configuration dictionary

        Args:
            config: Configuration dictionary

        Returns:
            CoordinateMapper instance
        """
        coord_config = config.get('coordinate_mapping', {})

        return cls(
            bmp_width=coord_config.get('bmp_width', 1024),
            bmp_height=coord_config.get('bmp_height', 5137),
            npy_width=coord_config.get('npy_width', 6400),
            npy_height=coord_config.get('npy_height', 5137),
            apply_y_flip=coord_config.get('apply_y_flip', True)
        )

    def __repr__(self) -> str:
        return (f"CoordinateMapper("
                f"BMP: {self.bmp_width}x{self.bmp_height}, "
                f"NPY: {self.npy_width}x{self.npy_height}, "
                f"Scale: {self.scale_x:.2f}x{self.scale_y:.2f}, "
                f"Flip: {self.apply_y_flip})")
