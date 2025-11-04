"""
Label Generation Module

Generates pixel-wise binary masks and instance segmentation labels
from mapped bounding box annotations.

Verified: 2025-11-04
Based on: scripts/mine_labeling/06_flip_bbox_mapping.py
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class LabelGenerator:
    """
    Generates label masks from bounding box annotations

    Supports:
    - Binary segmentation (0=background, 1=mine)
    - Instance segmentation (unique ID per mine)
    - Bounding box expansion/padding
    """

    def __init__(self,
                 height: int,
                 width: int,
                 background_value: int = 0,
                 mine_value: int = 1,
                 enable_instance_ids: bool = False,
                 bbox_padding: int = 0):
        """
        Initialize label generator

        Args:
            height: Label mask height
            width: Label mask width
            background_value: Background pixel value (default: 0)
            mine_value: Mine pixel value (default: 1)
            enable_instance_ids: Use unique ID per mine (default: False)
            bbox_padding: Pixels to expand bounding boxes (default: 0)
        """
        self.height = height
        self.width = width
        self.background_value = background_value
        self.mine_value = mine_value
        self.enable_instance_ids = enable_instance_ids
        self.bbox_padding = bbox_padding

    def create_binary_mask(self, annotations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create binary label mask (0=background, 1=mine)

        Args:
            annotations: List of mapped annotations with 'mapped_npy' bbox

        Returns:
            Binary mask array (H, W) dtype=uint8
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[:] = self.background_value

        for ann in annotations:
            bbox = ann['mapped_npy']

            # Apply padding
            xmin = max(0, bbox['xmin'] - self.bbox_padding)
            ymin = max(0, min(bbox['ymin'], bbox['ymax']) - self.bbox_padding)
            xmax = min(self.width, bbox['xmax'] + self.bbox_padding)
            ymax = min(self.height, max(bbox['ymin'], bbox['ymax']) + self.bbox_padding)

            # Fill region
            mask[ymin:ymax, xmin:xmax] = self.mine_value

        return mask

    def create_instance_mask(self, annotations: List[Dict[str, Any]],
                           instance_start_id: int = 1) -> np.ndarray:
        """
        Create instance segmentation mask (unique ID per mine)

        Args:
            annotations: List of mapped annotations with 'mapped_npy' bbox
            instance_start_id: Starting instance ID (default: 1)

        Returns:
            Instance mask array (H, W) dtype=uint16
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint16)
        mask[:] = self.background_value

        for i, ann in enumerate(annotations):
            instance_id = instance_start_id + i
            bbox = ann['mapped_npy']

            # Apply padding
            xmin = max(0, bbox['xmin'] - self.bbox_padding)
            ymin = max(0, min(bbox['ymin'], bbox['ymax']) - self.bbox_padding)
            xmax = min(self.width, bbox['xmax'] + self.bbox_padding)
            ymax = min(self.height, max(bbox['ymin'], bbox['ymax']) + self.bbox_padding)

            # Fill region with instance ID
            mask[ymin:ymax, xmin:xmax] = instance_id

        return mask

    def create_label_mask(self, annotations: List[Dict[str, Any]],
                         instance_start_id: int = 1) -> np.ndarray:
        """
        Create label mask (binary or instance based on configuration)

        Args:
            annotations: List of mapped annotations
            instance_start_id: Starting instance ID (default: 1)

        Returns:
            Label mask array
        """
        if self.enable_instance_ids:
            return self.create_instance_mask(annotations, instance_start_id)
        else:
            return self.create_binary_mask(annotations)

    def get_label_statistics(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate label mask statistics

        Args:
            mask: Label mask array

        Returns:
            Dictionary with statistics
        """
        total_pixels = mask.size
        mine_pixels = np.sum(mask > 0)
        background_pixels = total_pixels - mine_pixels

        stats = {
            'total_pixels': total_pixels,
            'mine_pixels': int(mine_pixels),
            'background_pixels': int(background_pixels),
            'mine_percentage': float(mine_pixels / total_pixels * 100),
            'background_percentage': float(background_pixels / total_pixels * 100)
        }

        # Instance-specific statistics
        if self.enable_instance_ids:
            unique_ids = np.unique(mask)
            num_instances = len(unique_ids) - 1  # Exclude background (0)

            stats['num_instances'] = num_instances
            stats['instance_ids'] = unique_ids[unique_ids > 0].tolist()

            # Per-instance pixel counts
            instance_sizes = {}
            for instance_id in stats['instance_ids']:
                instance_sizes[int(instance_id)] = int(np.sum(mask == instance_id))

            stats['instance_sizes'] = instance_sizes

        return stats

    def create_bbox_from_mask(self, mask: np.ndarray,
                             instance_id: Optional[int] = None) -> Dict[str, int]:
        """
        Extract bounding box from mask region

        Args:
            mask: Label mask
            instance_id: Specific instance ID (for instance masks)

        Returns:
            Bounding box dictionary
        """
        if instance_id is not None:
            # Instance-specific region
            region = (mask == instance_id)
        else:
            # All non-background regions
            region = (mask > 0)

        # Find bounding box
        rows = np.any(region, axis=1)
        cols = np.any(region, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Empty region
            return {
                'xmin': 0,
                'ymin': 0,
                'xmax': 0,
                'ymax': 0,
                'width': 0,
                'height': 0
            }

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return {
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax) + 1,
            'ymax': int(ymax) + 1,
            'width': int(xmax - xmin + 1),
            'height': int(ymax - ymin + 1)
        }

    def extract_all_bboxes(self, mask: np.ndarray) -> List[Dict[str, int]]:
        """
        Extract all bounding boxes from instance mask

        Args:
            mask: Instance segmentation mask

        Returns:
            List of bounding boxes
        """
        if not self.enable_instance_ids:
            # Single bbox for all mines
            return [self.create_bbox_from_mask(mask)]

        # Extract per-instance bboxes
        unique_ids = np.unique(mask)
        instance_ids = unique_ids[unique_ids > 0]

        bboxes = []
        for instance_id in instance_ids:
            bbox = self.create_bbox_from_mask(mask, instance_id)
            bbox['instance_id'] = int(instance_id)
            bboxes.append(bbox)

        return bboxes

    @classmethod
    def from_config(cls, config: Dict[str, Any],
                   height: int, width: int) -> 'LabelGenerator':
        """
        Create LabelGenerator from configuration dictionary

        Args:
            config: Configuration dictionary
            height: Label mask height
            width: Label mask width

        Returns:
            LabelGenerator instance
        """
        label_config = config.get('labeling', {})

        return cls(
            height=height,
            width=width,
            background_value=label_config.get('background_value', 0),
            mine_value=label_config.get('mine_value', 1),
            enable_instance_ids=label_config.get('enable_instance_ids', False),
            bbox_padding=label_config.get('bbox_padding', 0)
        )

    def __repr__(self) -> str:
        return (f"LabelGenerator("
                f"size: {self.width}x{self.height}, "
                f"instance: {self.enable_instance_ids}, "
                f"padding: {self.bbox_padding})")
