"""
Mine Sampler Module

Extracts mine samples from labeled NPY data for training.

Author: Mine Detection Team
Date: 2025-11-04
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class MineSampler:
    """
    Samples mine regions from labeled intensity data

    Features:
    - Extract fixed-size patches around mines
    - Center mines in patches
    - Handle boundary cases
    - Support instance-based sampling
    """

    def __init__(self,
                 patch_size: int = 128,
                 center_crop: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize mine sampler

        Args:
            patch_size: Size of extracted patches (default: 128)
            center_crop: Center mines in patches (default: True)
            random_seed: Random seed for reproducibility
        """
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def extract_mine_patch(self,
                          intensity: np.ndarray,
                          label_mask: np.ndarray,
                          bbox: Dict[str, int],
                          mine_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract single mine patch

        Args:
            intensity: Intensity array (H, W)
            label_mask: Label mask (H, W)
            bbox: Bounding box dictionary
            mine_id: Mine identifier

        Returns:
            Dictionary with patch data or None if extraction fails
        """
        height, width = intensity.shape

        if self.center_crop:
            # Center mine in patch
            center_x = (bbox['xmin'] + bbox['xmax']) // 2
            center_y = (bbox['ymin'] + bbox['ymax']) // 2

            half_size = self.patch_size // 2

            # Calculate patch bounds
            x1 = center_x - half_size
            y1 = center_y - half_size
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size

        else:
            # Use top-left corner of bbox
            x1 = bbox['xmin']
            y1 = bbox['ymin']
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            # Handle boundary case with padding
            pad_x1 = max(0, -x1)
            pad_y1 = max(0, -y1)
            pad_x2 = max(0, x2 - width)
            pad_y2 = max(0, y2 - height)

            # Clip to valid range
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)
            x2_clip = min(width, x2)
            y2_clip = min(height, y2)

            # Extract valid region
            intensity_crop = intensity[y1_clip:y2_clip, x1_clip:x2_clip]
            label_crop = label_mask[y1_clip:y2_clip, x1_clip:x2_clip]

            # Pad to patch_size
            intensity_patch = np.pad(
                intensity_crop,
                ((pad_y1, pad_y2), (pad_x1, pad_x2)),
                mode='edge'
            )
            label_patch = np.pad(
                label_crop,
                ((pad_y1, pad_y2), (pad_x1, pad_x2)),
                mode='constant',
                constant_values=0
            )

        else:
            # Direct extraction
            intensity_patch = intensity[y1:y2, x1:x2].copy()
            label_patch = label_mask[y1:y2, x1:x2].copy()

        # Verify patch size
        if intensity_patch.shape != (self.patch_size, self.patch_size):
            return None

        return {
            'mine_id': mine_id,
            'intensity': intensity_patch,
            'label': label_patch,
            'bbox_original': bbox,
            'patch_bounds': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            },
            'is_padded': (x1 < 0 or y1 < 0 or x2 > width or y2 > height)
        }

    def extract_all_mines(self,
                         intensity: np.ndarray,
                         label_mask: np.ndarray,
                         annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract all mine samples from data

        Args:
            intensity: Intensity array
            label_mask: Label mask
            annotations: List of annotations with 'mapped_npy' bbox

        Returns:
            List of extracted mine samples
        """
        samples = []

        for i, ann in enumerate(annotations):
            mine_id = i + 1
            bbox = ann['mapped_npy']

            sample = self.extract_mine_patch(
                intensity, label_mask, bbox, mine_id
            )

            if sample is not None:
                # Add annotation metadata
                sample['name'] = ann.get('name', f'mine_{mine_id}')
                samples.append(sample)

        return samples

    def get_sampling_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate sampling statistics

        Args:
            samples: List of extracted samples

        Returns:
            Dictionary with statistics
        """
        total_samples = len(samples)
        padded_samples = sum(1 for s in samples if s['is_padded'])

        # Calculate label coverage
        label_coverages = []
        for sample in samples:
            label_patch = sample['label']
            coverage = np.sum(label_patch > 0) / label_patch.size
            label_coverages.append(coverage)

        stats = {
            'total_samples': total_samples,
            'padded_samples': padded_samples,
            'patch_size': self.patch_size,
            'label_coverage': {
                'min': float(np.min(label_coverages)) if label_coverages else 0,
                'max': float(np.max(label_coverages)) if label_coverages else 0,
                'mean': float(np.mean(label_coverages)) if label_coverages else 0,
                'std': float(np.std(label_coverages)) if label_coverages else 0
            }
        }

        return stats

    def save_samples(self,
                    samples: List[Dict[str, Any]],
                    output_dir: Path,
                    format: str = 'npz') -> None:
        """
        Save extracted samples to disk

        Args:
            samples: List of extracted samples
            output_dir: Output directory
            format: 'npz' or 'npy' (default: 'npz')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            mine_id = sample['mine_id']

            if format == 'npz':
                # Save as NPZ with metadata
                output_path = output_dir / f'mine_{mine_id:03d}.npz'

                np.savez_compressed(
                    output_path,
                    intensity=sample['intensity'],
                    label=sample['label'],
                    bbox=np.array([
                        sample['bbox_original']['xmin'],
                        sample['bbox_original']['ymin'],
                        sample['bbox_original']['xmax'],
                        sample['bbox_original']['ymax']
                    ]),
                    is_padded=sample['is_padded']
                )

            elif format == 'npy':
                # Save intensity and label separately
                intensity_path = output_dir / f'mine_{mine_id:03d}_intensity.npy'
                label_path = output_dir / f'mine_{mine_id:03d}_label.npy'

                np.save(intensity_path, sample['intensity'])
                np.save(label_path, sample['label'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MineSampler':
        """
        Create MineSampler from configuration

        Args:
            config: Configuration dictionary

        Returns:
            MineSampler instance
        """
        sampling_config = config.get('sampling', {})

        return cls(
            patch_size=sampling_config.get('patch_size', 128),
            center_crop=sampling_config.get('center_crop', True),
            random_seed=sampling_config.get('random_seed', None)
        )

    def __repr__(self) -> str:
        return (f"MineSampler("
                f"patch_size={self.patch_size}, "
                f"center_crop={self.center_crop})")
